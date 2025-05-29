import os
import imaplib
import email
import smtplib
import pandas as pd
import openai
import streamlit as st
from datetime import datetime, timedelta
import re
import json
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
import html
import time
import io

# Google Drive & OAuth imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

# Configuration
IMAP_SERVER, SMTP_SERVER = 'imap.gmail.com', 'smtp.gmail.com'
CSV_FILE, LOG_FILE = 'emails.csv', 'app.log'
CUSTOM_PROMPT_FILE, DRIVE_CREDS_FILE, DRIVE_TOKEN_FILE = (
    'custom_prompt.json',
    'drive_credentials.json',
    'drive_token.json'
)
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# OpenAI API Key from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY
except KeyError:
    st.error("❌ Please add OPENAI_API_KEY to your Streamlit secrets")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading OpenAI API key: {str(e)}")
    st.stop()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def log(msg, level="info"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    getattr(logging, level)(msg)

# Streamlit configuration
st.set_page_config(
    page_title="AI Email Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center; }
    .custom-prompt-box { background: #f0f8ff; border: 2px solid #4a90e2; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin: 0.5rem 0; }
    </style>
    """, unsafe_allow_html=True)

# Gmail Functions
def login_gmail(user, password, max_retries=3):
    for attempt in range(max_retries):
        try:
            mail = imaplib.IMAP4_SSL(IMAP_SERVER, 993)
            mail.sock.settimeout(60)
            mail.login(user, password)
            log(f"Successfully logged in to Gmail for {user}")
            return mail
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Gmail login failed: {str(e)}")
            log(f"Login attempt {attempt + 1} failed, retrying: {str(e)}", "warning")
            time.sleep(2)

def clean_html_text(text):
    if not text: return ""
    try:
        text = html.unescape(text)
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            for script in soup(["script", "style"]): script.decompose()
            text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()
    except:
        return str(text) if text else ""

def get_body(msg):
    try:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    body = part.get_payload(decode=True).decode(errors='ignore')
                    break
                elif content_type == 'text/html' and not body:
                    body = clean_html_text(part.get_payload(decode=True).decode(errors='ignore'))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors='ignore')
                if msg.get_content_type() == 'text/html':
                    body = clean_html_text(body)
        return clean_html_text(body)
    except Exception as e:
        log(f"Error extracting email body: {str(e)}", "error")
        return ""

def fetch_emails(mail, folder='inbox', limit=20, comprehensive=False, progress_callback=None):
    try:
        folder_name = folder if folder != 'sent' else '"[Gmail]/Sent Mail"'
        mail.select(folder_name)
        
        _, messages = mail.search(None, 'ALL')
        email_ids = messages[0].split() if messages[0] else []
        if not email_ids: return []
        if not comprehensive:
            email_ids = email_ids[-limit:]
        
        emails, failed_count = [], 0
        batch_size = 25 if comprehensive else 50
        
        for i in range(0, len(email_ids), batch_size):
            batch = email_ids[i:i + batch_size]
            if progress_callback:
                progress = (i / len(email_ids)) * 100
                progress_callback(progress, f"Processing {folder} batch {i//batch_size + 1}")
            for eid in batch:
                try:
                    mail.sock.settimeout(30)
                    _, data = mail.fetch(eid, '(RFC822)')
                    if not data or not data[0] or not data[0][1]:
                        failed_count += 1
                        continue
                    msg = email.message_from_bytes(data[0][1])
                    email_date = msg.get('Date', '')
                    try:
                        from email.utils import parsedate_to_datetime
                        parsed_date = parsedate_to_datetime(email_date)
                    except:
                        parsed_date = datetime.now()
                    emails.append({
                        'id': eid.decode(),
                        'subject': str(msg.get('Subject', ''))[:200],
                        'from': str(msg.get('From', ''))[:100],
                        'to': str(msg.get('To', ''))[:100],
                        'date': email_date,
                        'parsed_date': parsed_date,
                        'body': get_body(msg)[:3000 if comprehensive else 2000]
                    })
                except Exception as e:
                    failed_count += 1
                    if "SSL" in str(e) and failed_count > 10:
                        break
                    continue
            if i + batch_size < len(email_ids):
                time.sleep(1 if comprehensive else 0.5)
        
        log(f"Fetched {len(emails)} emails from {folder} ({failed_count} failed)")
        return emails
    except Exception as e:
        log(f"Error fetching emails: {str(e)}", "error")
        return []

def match_replies(inbox_emails, sent_emails):
    pairs = []
    for inbox_email in inbox_emails:
        try:
            customer_from = str(inbox_email.get('from', '')).lower()
            customer_email = re.search(r'<(.+?)>', customer_from)
            customer_email = customer_email.group(1) if customer_email else customer_from
            reply = ''
            for sent_email in sent_emails:
                if customer_email in str(sent_email.get('to', '')).lower():
                    reply = sent_email.get('body', '')
                    break
            pairs.append({
                'subject': inbox_email.get('subject', ''),
                'from': inbox_email.get('from', ''),
                'date': inbox_email.get('date', ''),
                'parsed_date': inbox_email.get('parsed_date', datetime.now()),
                'customer': str(inbox_email.get('body', '')).strip(),
                'reply': str(reply).strip(),
                'has_reply': bool(str(reply).strip())
            })
        except:
            continue
    log(f"Matched {len(pairs)} email pairs")
    return pairs

def send_email(user, password, to, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'] = user, to
        msg['Subject'] = f"Re: {subject}" if not subject.lower().startswith('re:') else subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        with smtplib.SMTP(SMTP_SERVER, 587) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        log(f"Email sent successfully to {to}")
        return True, "Email sent successfully"
    except Exception as e:
        log(f"Error sending email: {str(e)}", "error")
        return False, f"Failed to send email: {str(e)}"

# Knowledge Base Functions
def extract_keywords(text):
    if not text: return []
    patterns = [
        r'\b(refund|return|cancel|exchange)\b',
        r'\b(order|purchase|payment|billing)\b',
        r'\b(shipping|delivery|tracking)\b',
        r'\b(problem|issue|help|support)\b',
        r'\b(account|login|password)\b',
        r'\b(complaint|feedback|review)\b'
    ]
    keywords = []
    for pattern in patterns:
        keywords.extend(re.findall(pattern, str(text).lower(), re.IGNORECASE))
    return list(set(keywords))

def load_custom_prompt():
    try:
        if os.path.exists(CUSTOM_PROMPT_FILE):
            with open(CUSTOM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('custom_prompt', ''), data.get('enabled', False)
        return '', False
    except:
        return '', False

def save_custom_prompt(custom_prompt, enabled):
    try:
        data = {
            'custom_prompt': custom_prompt,
            'enabled': enabled,
            'updated_at': datetime.now().isoformat()
        }
        with open(CUSTOM_PROMPT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

# Google Drive Functions - Updated for Streamlit Cloud
def setup_google_drive_auth():
    if not GOOGLE_DRIVE_AVAILABLE:
        return None, "Google Drive libraries not installed"
    try:
        creds = None
        if os.path.exists(DRIVE_TOKEN_FILE):
            with open(DRIVE_TOKEN_FILE, 'r') as f:
                creds = Credentials.from_authorized_user_info(json.load(f), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(DRIVE_TOKEN_FILE, 'w') as f:
                    f.write(creds.to_json())
            else:
                return None, "Google Drive authentication required"
        return creds, "Success"
    except Exception as e:
        return None, f"Authentication failed: {str(e)}"

def get_drive_filename(email_user, filename):
    """Generate the Google Drive filename based on email user prefix"""
    email_prefix = email_user.split('@')[0] if '@' in email_user else email_user
    return f"{email_prefix}_{filename}"

def ensure_knowledge_bases_folder():
    """Create or find the 'knowledge bases' folder on Google Drive"""
    try:
        creds, message = setup_google_drive_auth()
        if not creds:
            return None, message
        
        service = build('drive', 'v3', credentials=creds)
        
        # Search for existing 'knowledge bases' folder
        results = service.files().list(
            q="name='knowledge bases' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id,name)"
        ).execute()
        
        files = results.get('files', [])
        
        if files:
            # Folder exists, return its ID
            folder_id = files[0]['id']
            log("Found existing 'knowledge bases' folder on Google Drive")
            return folder_id, "Folder found"
        else:
            # Create the folder
            folder_metadata = {
                'name': 'knowledge bases',
                'mimeType': 'application/vnd.google-apps.folder',
                'description': 'AI Email Assistant Knowledge Bases Storage'
            }
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder.get('id')
            log("Created 'knowledge bases' folder on Google Drive")
            return folder_id, "Folder created"
    
    except Exception as e:
        log(f"Error managing knowledge bases folder: {str(e)}", "error")
        return None, f"Folder management failed: {str(e)}"

def find_existing_kb_file(email_user, folder_id):
    """Find existing KB file for a user in the knowledge bases folder"""
    try:
        creds, message = setup_google_drive_auth()
        if not creds:
            return None, message
        
        service = build('drive', 'v3', credentials=creds)
        drive_filename = get_drive_filename(email_user, 'knowledge_base.json')
        
        # Search for existing KB file in the specific folder
        results = service.files().list(
            q=f"name='{drive_filename}' and parents='{folder_id}' and trashed=false",
            fields="files(id,name,modifiedTime)"
        ).execute()
        
        files = results.get('files', [])
        return files[0] if files else None, "Success"
    
    except Exception as e:
        log(f"Error finding existing KB file: {str(e)}", "error")
        return None, f"Search failed: {str(e)}"

def load_kb_from_drive(email_user):
    """Load knowledge base directly from Google Drive"""
    if not GOOGLE_DRIVE_AVAILABLE:
        log("Google Drive not available for KB loading", "warning")
        return {'total': 0, 'with_replies': 0, 'emails': [], 'user_email': email_user or 'unknown'}
    
    if not email_user:
        log("No email user provided for KB loading", "warning")
        return {'total': 0, 'with_replies': 0, 'emails': [], 'user_email': 'unknown'}
    
    try:
        # Ensure knowledge bases folder exists
        folder_id, folder_message = ensure_knowledge_bases_folder()
        if not folder_id:
            log(f"Failed to access knowledge bases folder: {folder_message}", "warning")
            return {'total': 0, 'with_replies': 0, 'emails': [], 'user_email': email_user}
        
        # Find existing KB file
        existing_file, search_message = find_existing_kb_file(email_user, folder_id)
        if not existing_file:
            log(f"No knowledge base found on Google Drive for {email_user}", "info")
            return {'total': 0, 'with_replies': 0, 'emails': [], 'user_email': email_user}
        
        # Download and parse the knowledge base
        creds, _ = setup_google_drive_auth()
        service = build('drive', 'v3', credentials=creds)
        file_content = service.files().get_media(fileId=existing_file['id']).execute()
        content_str = file_content.decode('utf-8')
        kb_data = json.loads(content_str)
        
        log(f"Successfully loaded KB from Google Drive: {kb_data.get('total', 0)} emails")
        return kb_data
        
    except Exception as e:
        log(f"Error loading KB from Google Drive: {str(e)}", "error")
        return {'total': 0, 'with_replies': 0, 'emails': [], 'user_email': email_user or 'unknown'}

def save_kb_to_drive(kb_data, email_user):
    """Save knowledge base directly to Google Drive in organized folder structure"""
    if not GOOGLE_DRIVE_AVAILABLE:
        return False, "Google Drive not available"
    
    if not email_user:
        return False, "No email user provided"
    
    try:
        # Ensure knowledge bases folder exists
        folder_id, folder_message = ensure_knowledge_bases_folder()
        if not folder_id:
            return False, f"Failed to access knowledge bases folder: {folder_message}"
        
        creds, message = setup_google_drive_auth()
        if not creds:
            return False, message
        
        service = build('drive', 'v3', credentials=creds)
        drive_filename = get_drive_filename(email_user, 'knowledge_base.json')
        
        # Check if file already exists in the folder
        existing_file, search_message = find_existing_kb_file(email_user, folder_id)
        
        # Prepare file content
        file_stream = io.BytesIO(json.dumps(kb_data, indent=2, ensure_ascii=False).encode('utf-8'))
        media = MediaIoBaseUpload(file_stream, mimetype='application/json', resumable=True)
        
        if existing_file:
            # Update existing file (prevents duplicates)
            service.files().update(fileId=existing_file['id'], media_body=media).execute()
            log(f"Updated existing knowledge base on Google Drive: {drive_filename}")
            action = "updated"
        else:
            # Create new file in the knowledge bases folder
            file_metadata = {
                'name': drive_filename,
                'parents': [folder_id],  # Place in knowledge bases folder
                'description': f'AI Email Assistant KB for {email_user} - Created: {datetime.now().isoformat()}'
            }
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            log(f"Created new knowledge base on Google Drive: {drive_filename}")
            action = "created"
        
        return True, f"Knowledge base {action} in Google Drive folder 'knowledge bases' as '{drive_filename}'"
        
    except Exception as e:
        log(f"Google Drive KB save error: {str(e)}", "error")
        return False, f"Save failed: {str(e)}"

def generate_reply(query, email_user):
    """Generate reply using KB loaded directly from Google Drive"""
    try:
        if not query.strip(): 
            return "Error: Empty query provided"
        
        # Load KB directly from Google Drive
        kb = load_kb_from_drive(email_user)
        
        custom_prompt, custom_enabled = load_custom_prompt()
        query_keywords = set(extract_keywords(query))
        similar = []
        
        for email in kb.get('emails', []):
            if email.get('has_reply'):
                email_keywords = set(email.get('keywords', []))
                score = len(query_keywords.intersection(email_keywords))
                if score > 0:
                    similar.append((email, score))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        similar = [item[0] for item in similar[:3]]
        
        context = "No similar examples found." if not similar else ""
        for i, email in enumerate(similar):
            customer_text = str(email.get('customer', ''))[:150]
            reply_text = str(email.get('reply', ''))[:150]
            context += f"Example {i+1}:\nCustomer: {customer_text}...\nReply: {reply_text}...\n\n"
        
        base_prompt = "You are a professional customer service agent. Generate helpful, concise, and polite replies based on the examples provided. Keep responses under 200 words."
        system_prompt = f"{base_prompt}\n\nAdditional Instructions: {custom_prompt.strip()}" if custom_enabled and custom_prompt.strip() else base_prompt
        
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}New Customer Email: {query[:500]}\n\nGenerate a professional reply:"}
            ],
            max_tokens=250, temperature=0.7, timeout=30
        )
        return response.choices[0].message.content.strip()
        
    except openai.error.RateLimitError:
        return "I apologize for the delay. We are experiencing high volume. Please allow us some time to respond to your inquiry personally."
    except Exception as e:
        log(f"Error generating reply: {str(e)}", "error")
        return "Thank you for contacting us. We will review your message and respond as soon as possible."

def create_knowledge_base(email_pairs, email_user=None, comprehensive=False):
    """Create knowledge base and save directly to Google Drive"""
    try:
        kb = {
            'total': len(email_pairs),
            'with_replies': 0,
            'emails': [],
            'created_at': datetime.now().isoformat(),
            'user_email': email_user or 'unknown',
            'comprehensive': comprehensive
        }
        
        if comprehensive:
            kb['statistics'] = {
                'total_processed': len(email_pairs),
                'with_replies': 0,
                'without_replies': 0,
                'keywords_extracted': 0,
                'date_range': {'earliest': None, 'latest': None}
            }
            earliest_date = latest_date = None
            total_keywords = 0
        
        for pair in email_pairs:
            try:
                has_reply = bool(pair.get('reply', '').strip())
                if has_reply: 
                    kb['with_replies'] += 1
                
                keywords = extract_keywords(pair.get('customer', ''))
                email_data = {
                    'customer': str(pair.get('customer', ''))[:2000 if comprehensive else 1000],
                    'reply': str(pair.get('reply', ''))[:2000 if comprehensive else 1000] if has_reply else None,
                    'has_reply': has_reply,
                    'keywords': keywords
                }
                
                if comprehensive:
                    kb['statistics']['with_replies' if has_reply else 'without_replies'] += 1
                    total_keywords += len(keywords)
                    email_date = pair.get('parsed_date')
                    if email_date:
                        if not earliest_date or email_date < earliest_date:
                            earliest_date = email_date
                        if not latest_date or email_date > latest_date:
                            latest_date = email_date
                    email_data.update({
                        'subject': str(pair.get('subject', ''))[:200],
                        'from': str(pair.get('from', ''))[:100],
                        'date': pair.get('date', ''),
                        'parsed_date': email_date.isoformat() if email_date else None
                    })
                
                kb['emails'].append(email_data)
            except:
                continue
        
        if comprehensive:
            kb['statistics']['keywords_extracted'] = total_keywords
            if earliest_date:
                kb['statistics']['date_range']['earliest'] = earliest_date.isoformat()
            if latest_date:
                kb['statistics']['date_range']['latest'] = latest_date.isoformat()
        
        # Save directly to Google Drive
        if email_user and GOOGLE_DRIVE_AVAILABLE:
            try:
                backup_success, backup_message = save_kb_to_drive(kb, email_user)
                kb['last_backup'] = datetime.now().isoformat()
                kb['backup_status'] = 'success' if backup_success else 'failed'
                if not backup_success:
                    kb['backup_error'] = backup_message
                # Update the saved version with backup status
                if backup_success:
                    save_kb_to_drive(kb, email_user)
            except Exception as e:
                kb['backup_status'] = 'error'
                kb['backup_error'] = str(e)
        
        log(f"Created {'comprehensive' if comprehensive else 'standard'} knowledge base with {kb['total']} emails")
        return kb
        
    except Exception as e:
        log(f"Error creating knowledge base: {str(e)}", "error")
        return {'total': 0, 'with_replies': 0, 'emails': [], 'user_email': email_user or 'unknown'}

# Storage Functions (for CSV and other local files, not KB)
def save_data(data, file):
    try:
        if file.endswith('.csv'):
            if not data:
                pd.DataFrame(columns=['subject','from','date','customer','reply','has_reply']).to_csv(file, index=False)
            else:
                pd.DataFrame(data).to_csv(file, index=False)
        else:
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log(f"Error saving data to {file}: {str(e)}", "error")
        return False

def load_data(file):
    try:
        if not os.path.exists(file):
            return [] if file.endswith('.csv') else {}
        if file.endswith('.csv'):
            df = pd.read_csv(file)
            return df.fillna('').to_dict('records') if not df.empty else []
        else:
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
    except Exception as e:
        log(f"Error loading {file}: {str(e)}", "error")
        return [] if file.endswith('.csv') else {}

# Legacy Google Drive functions (updated for folder structure)
def upload_to_google_drive(file_content, filename, email_user):
    """Legacy function - updated to use folder structure for KB files"""
    if filename == 'knowledge_base.json':
        return save_kb_to_drive(file_content, email_user)
    
    # For other files, keep original functionality but also use folder structure
    if not GOOGLE_DRIVE_AVAILABLE:
        return False, "Google Drive not available"
    try:
        # Ensure knowledge bases folder exists for consistency
        folder_id, folder_message = ensure_knowledge_bases_folder()
        if not folder_id:
            log(f"Warning: Could not access knowledge bases folder: {folder_message}", "warning")
            # Continue without folder structure for non-KB files
        
        creds, message = setup_google_drive_auth()
        if not creds:
            return False, message
        
        service = build('drive', 'v3', credentials=creds)
        drive_filename = get_drive_filename(email_user, filename)
        
        # Search for existing file (in folder if available)
        if folder_id:
            query = f"name='{drive_filename}' and parents='{folder_id}' and trashed=false"
        else:
            query = f"name='{drive_filename}' and trashed=false"
        
        results = service.files().list(q=query, fields="files(id,name)").execute()
        files = results.get('files', [])
        
        file_stream = io.BytesIO(json.dumps(file_content, indent=2, ensure_ascii=False).encode('utf-8'))
        media = MediaIoBaseUpload(file_stream, mimetype='application/json', resumable=True)
        
        if files:
            # Update existing file
            service.files().update(fileId=files[0]['id'], media_body=media).execute()
            log(f"Updated file on Google Drive: {drive_filename}")
            action = "updated"
        else:
            # Create new file
            file_metadata = {
                'name': drive_filename, 
                'description': f'AI Email Assistant file for {email_user}'
            }
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            log(f"Uploaded new file to Google Drive: {drive_filename}")
            action = "created"
        
        location = "in knowledge bases folder" if folder_id else "in root"
        return True, f"File {action} on Google Drive {location} as '{drive_filename}'"
        
    except Exception as e:
        log(f"Google Drive upload error: {str(e)}", "error")
        return False, f"Upload failed: {str(e)}"

def auto_backup_to_drive(kb_data, email_user):
    """Legacy function - now redirects to save_kb_to_drive"""
    if not email_user or not kb_data:
        return False, "Google Drive not configured"
    return save_kb_to_drive(kb_data, email_user)

# Google Drive OAuth2 for Streamlit Cloud (No Flask)
def streamlit_google_oauth():
    """Handle Google OAuth2 authentication without Flask"""
    st.subheader("🔐 Google Drive Authorization")
    
    if not os.path.exists(DRIVE_CREDS_FILE):
        st.error("❌ Credentials file not found. Please upload first.")
        return False
    
    try:
        # Create flow with OOB redirect URI for Streamlit Cloud
        flow = Flow.from_client_secrets_file(
            DRIVE_CREDS_FILE,
            scopes=SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # Out-of-band flow
        )
        
        # Get authorization URL
        auth_url, _ = flow.authorization_url(prompt='consent')
        
        st.markdown(f"""
        **Step 1:** Click the link below to authorize Google Drive access:
        
        🔗 [**Authorize Google Drive Access**]({auth_url})
        
        **Step 2:** After authorization, copy the code from Google and paste it below:
        """)
        
        # Input field for authorization code
        auth_code = st.text_input("📝 Paste the authorization code here:", type="password")
        
        if st.button("✅ Complete Authorization") and auth_code:
            try:
                # Exchange code for credentials
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                
                # Save credentials
                with open(DRIVE_TOKEN_FILE, 'w') as token_file:
                    token_file.write(creds.to_json())
                
                st.success("✅ Google Drive authorization successful!")
                st.balloons()
                time.sleep(2)
                st.rerun()
                return True
                
            except Exception as e:
                st.error(f"❌ Authorization failed: {str(e)}")
                return False
        
        return False
        
    except Exception as e:
        st.error(f"❌ OAuth setup failed: {str(e)}")
        return False

# UI Functions
def display_metrics(kb_data):
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("📧 Total Emails", kb_data.get('total', 0)),
        ("✅ With Replies", kb_data.get('with_replies', 0)),
        ("⏳ Pending", kb_data.get('total', 0) - kb_data.get('with_replies', 0)),
        ("🧠 KB Size", len(kb_data.get('emails', [])))
    ]
    for col, (title, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f'<div class="metric-card"><h3>{title}</h3><h2>{value}</h2></div>', unsafe_allow_html=True)

def comprehensive_email_fetch(user, password, progress_callback=None):
    try:
        mail = login_gmail(user, password)
        if progress_callback:
            progress_callback(10, "Fetching inbox emails...")
        inbox_emails = fetch_emails(
            mail, 'inbox', comprehensive=True,
            progress_callback=lambda p, m: progress_callback(10 + p*0.4, f"Inbox: {m}") if progress_callback else None
        )
        if progress_callback:
            progress_callback(50, "Fetching sent emails...")
        sent_emails = fetch_emails(
            mail, 'sent', comprehensive=True,
            progress_callback=lambda p, m: progress_callback(50 + p*0.4, f"Sent: {m}") if progress_callback else None
        )
        mail.logout()
        total = len(inbox_emails) + len(sent_emails)
        log(f"Comprehensive fetch: {len(inbox_emails)} inbox + {len(sent_emails)} sent = {total} total")
        return inbox_emails, sent_emails
    except Exception as e:
        log(f"Comprehensive fetch error: {str(e)}", "error")
        raise

def search_and_load_emails(email_user, email_pass, search_type, search_value, date_from, date_to, email_limit, mode="standard"):
    try:
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            if mode == "comprehensive":
                st.info("🚀 Fetching ALL emails (may take 30+ minutes for large accounts)")
                def update_progress(percent, message):
                    progress_bar.progress(percent / 100)
                    status_text.text(f"🔄 {message}")
                inbox_emails, sent_emails = comprehensive_email_fetch(
                    email_user, email_pass, update_progress
                )
                total_fetched = len(inbox_emails) + len(sent_emails)
                st.success(f"📊 Fetched {total_fetched} total emails ({len(inbox_emails)} inbox + {len(sent_emails)} sent)")
            else:
                st.info(f"📊 Fetching up to {email_limit} emails per folder")
                status_text.text("🔐 Authenticating with Gmail...")
                progress_bar.progress(20)
                mail = login_gmail(email_user, email_pass)
                inbox_emails = fetch_emails(mail, 'inbox', email_limit)
                progress_bar.progress(60)
                sent_emails = fetch_emails(mail, 'sent', email_limit)
                mail.logout()
            
            status_text.text("🔄 Processing and matching emails...")
            progress_bar.progress(80)
            pairs = match_replies(inbox_emails, sent_emails)
            
            status_text.text("☁️ Saving knowledge base to Google Drive...")
            progress_bar.progress(90)
            
            if save_data(pairs, CSV_FILE):
                kb = create_knowledge_base(pairs, email_user, mode == "comprehensive")
                kb['last_updated'] = datetime.now().isoformat()
                kb['build_type'] = mode
                if mode == "comprehensive":
                    kb['total_fetched'] = len(inbox_emails) + len(sent_emails)
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                if mode == "comprehensive":
                    st.success(f"🎉 Comprehensive Knowledge Base Built! ({len(pairs)} conversations)")
                    if 'statistics' in kb:
                        stats = kb['statistics']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("📧 Total Emails", len(inbox_emails) + len(sent_emails))
                        with col2: st.metric("🔗 Conversations", len(pairs))
                        with col3: st.metric("✅ With Replies", stats.get('with_replies', 0))
                        with col4:
                            reply_rate = (stats.get('with_replies', 0) / len(pairs) * 100) if pairs else 0
                            st.metric("📈 Reply Rate", f"{reply_rate:.1f}%")
                else:
                    st.success(f"🎉 Successfully loaded {len(pairs)} emails!")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("📧 Inbox", len(inbox_emails))
                    with col2: st.metric("📤 Sent", len(sent_emails))
                    with col3: st.metric("🔗 Pairs", len(pairs))
                
                if 'backup_status' in kb:
                    if kb['backup_status'] == 'success':
                        st.success("☁️ Saved to Google Drive!")
                    else:
                        st.warning(f"⚠️ Google Drive save failed: {kb.get('backup_error', 'Unknown')}")
                
                display_metrics(kb)
                st.session_state.update({'last_search_results': pairs, 'current_kb': kb})
                time.sleep(2)
                progress_container.empty()
                st.rerun()
            else:
                st.error("❌ Failed to save email data")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        log(f"Search and load error: {str(e)}", "error")

def reply_management_section(email_user, email_pass):
    emails = load_data(CSV_FILE)
    if not emails:
        st.info("📭 No emails loaded. Please use the 'Search & Load' tab first.")
        return
    
    # Load KB from Google Drive for metrics display
    kb = load_kb_from_drive(email_user) if email_user else {'total': 0, 'with_replies': 0, 'emails': []}
    
    st.subheader("🔍 Find Emails to Reply")
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_text = st.text_input("🔍 Search:", placeholder="sender, subject, content...")
    with col2:
        filter_date_from = st.date_input("📅 From:", value=None, key="reply_filter_from")
    with col3:
        filter_date_to = st.date_input("📅 To:", value=None, key="reply_filter_to")

    pending = [e for e in emails if not e.get('has_reply', False) and str(e.get('customer', '')).strip()]
    if filter_text or filter_date_from or filter_date_to:
        filtered = []
        for email in pending:
            if filter_text and filter_text.lower() not in f"{email.get('subject','')} {email.get('from','')} {email.get('customer','')}".lower():
                continue
            if filter_date_from or filter_date_to:
                email_date = email.get('parsed_date')
                if email_date:
                    email_date_obj = email_date.date() if hasattr(email_date, 'date') else email_date
                    if filter_date_from and email_date_obj < filter_date_from: continue
                    if filter_date_to and email_date_obj > filter_date_to: continue
            filtered.append(email)
        pending = filtered
    
    if not pending:
        st.success("🎉 All emails have been replied to!" if not (filter_text or filter_date_from or filter_date_to) else "🔍 No pending emails match your filters.")
        return

    st.write(f"📬 Found **{len(pending)}** emails needing replies")
    if 'selected_emails' not in st.session_state:
        st.session_state.selected_emails = []

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Select All"):
            st.session_state.selected_emails = pending.copy()
            st.rerun()
    with col2:
        if st.button("❌ Clear Selection"):
            st.session_state.selected_emails = []
            st.rerun()
    with col3:
        st.write(f"Selected: **{len(st.session_state.selected_emails)}**")

    for i, email in enumerate(pending[:20]):
        col1, col2 = st.columns([0.5, 9.5])
        with col1:
            selected = st.checkbox("Select email", key=f"select_{i}", label_visibility="collapsed")
            if selected and email not in st.session_state.selected_emails:
                st.session_state.selected_emails.append(email)
            elif not selected and email in st.session_state.selected_emails:
                st.session_state.selected_emails.remove(email)
        with col2:
            from_field = str(email.get('from', 'Unknown'))[:50]
            subject_field = str(email.get('subject', 'No Subject'))[:60]
            with st.expander(f"📧 **{from_field}** | {subject_field}"):
                col_info, col_preview = st.columns([1, 2])
                with col_info:
                    st.write(f"**From:** {from_field}")
                    st.write(f"**Subject:** {subject_field}")
                    st.write(f"**Date:** {str(email.get('date',''))[:25]}")
                    keywords = extract_keywords(str(email.get('customer','')))
                    if any(word in ['urgent','asap','emergency'] for word in keywords):
                        st.error("🚨 High Priority")
                    elif any(word in ['refund','cancel','complaint'] for word in keywords):
                        st.warning("⚠️ Medium Priority")
                    else:
                        st.info("📝 Normal Priority")
                with col_preview:
                    customer_text = str(email.get('customer',''))
                    preview_text = customer_text[:300] + ("..." if len(customer_text) > 300 else "")
                    st.text_area(
                        "Email Content",
                        preview_text,
                        height=100,
                        disabled=True,
                        key=f"preview_{i}",
                        label_visibility="collapsed"
                    )
    
    if len(pending) > 20:
        st.info(f"📄 Showing 20 of {len(pending)} emails. Use filters to narrow results.")
    
    if st.session_state.selected_emails:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🤖 Generate {len(st.session_state.get('selected_emails',[]))} Replies", type="primary", use_container_width=True):
                generate_ai_replies(st.session_state.selected_emails, email_user)
        with col2:
            if st.button("🗑️ Clear Selected", use_container_width=True):
                st.session_state.selected_emails = []
                st.rerun()

    if 'generated_replies' in st.session_state and st.session_state.generated_replies:
        display_generated_replies(email_user, email_pass)

def generate_ai_replies(selected_emails, email_user):
    try:
        # Test if OpenAI API key is properly configured
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your-api-key"):
            st.error("❌ OpenAI API key not properly configured in Streamlit secrets")
            return
    except:
        st.error("❌ OpenAI API key not found in Streamlit secrets")
        return
    
    st.session_state.generated_replies = []
    with st.container():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, email in enumerate(selected_emails):
            status_text.text(f"🤖 Generating reply {idx+1} of {len(selected_emails)}...")
            customer_text = str(email.get('customer','')).strip()
            if customer_text:
                # Use the updated generate_reply function that loads KB from Drive
                reply = generate_reply(customer_text, email_user)
                st.session_state.generated_replies.append({'email':email,'draft':reply,'status':'draft'})
            progress_bar.progress((idx+1)/len(selected_emails))
        
        status_text.text("✅ Reply generation complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    st.success(f"🎉 Generated {len(st.session_state.generated_replies)} replies!")
    st.rerun()

def display_generated_replies(email_user, email_pass):
    st.divider()
    st.subheader("📝 Generated Replies - Review & Send")
    
    for i, reply_item in enumerate(st.session_state.generated_replies):
        email = reply_item['email']
        draft = reply_item['draft']
        status = reply_item.get('status','draft')
        status_icon = "✅" if status=='sent' else "❌" if status=='error' else "📝"
        subject = str(email.get('subject','No Subject'))[:50]
        
        with st.expander(f"{status_icon} Reply {i+1}: {subject}...", expanded=(i<3)):
            col1, col2 = st.columns([1,2])
            with col1:
                st.write(f"**To:** {str(email.get('from','Unknown'))[:40]}...")
                st.write(f"**Subject:** {subject}...")
                st.write(f"**Status:** {status.title()}")
                recipient_match = re.search(r'<(.+?)>', str(email.get('from','')))
                recipient_email = recipient_match.group(1) if recipient_match else str(email.get('from',''))
            with col2:
                st.write("**Original Customer Email:**")
                original_text = str(email.get('customer',''))[:300]
                st.text_area(
                    "Original Email",
                    original_text + ("..." if len(str(email.get('customer','')))>300 else ""),
                    height=80,
                    disabled=True,
                    key=f"orig_{i}",
                    label_visibility="collapsed"
                )
            
            st.write("**Generated Reply:**")
            edited_reply = st.text_area("Edit your reply:", draft, height=120, key=f"reply_edit_{i}")
            
            if status!='sent':
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if st.button("📧 Send Email", key=f"send_{i}", type="primary"):
                        success, message = send_email(email_user, email_pass, recipient_email, subject, edited_reply)
                        if success:
                            st.success(f"✅ {message}")
                            st.session_state.generated_replies[i]['status']='sent'
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                with c2:
                    if st.button("🔄 Regenerate", key=f"regen_{i}"):
                        # Use updated generate_reply function that loads KB from Drive
                        new_reply = generate_reply(str(email.get('customer','')).strip(), email_user)
                        st.session_state.generated_replies[i]['draft']=new_reply
                        st.session_state.generated_replies[i]['status']='draft'
                        st.success("✅ Reply regenerated!")
                        st.rerun()
                with c3:
                    if st.button("💾 Save Draft", key=f"save_{i}"):
                        drafts = load_data('drafts.json') if os.path.exists('drafts.json') else []
                        drafts.append({
                            'subject':subject,
                            'from':str(email.get('from','')),
                            'reply':edited_reply,
                            'saved_at':datetime.now().isoformat(),
                            'original_email':original_text
                        })
                        if save_data(drafts,'drafts.json'):
                            st.success("💾 Draft saved!")
                        else:
                            st.error("❌ Failed to save draft")
                with c4:
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.generated_replies.pop(i)
                        st.success("🗑️ Reply removed!")
                        st.rerun()
            else:
                st.success("✅ Email sent successfully!")
    
    if st.session_state.generated_replies:
        st.divider()
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("📧 Send All Replies", use_container_width=True):
                send_all_replies(email_user, email_pass)
        with b2:
            if st.button("💾 Save All Drafts", use_container_width=True):
                save_all_drafts()
        with b3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.generated_replies=[]
                st.rerun()

def send_all_replies(email_user, email_pass):
    pending_replies = [r for r in st.session_state.generated_replies if r.get('status')!='sent']
    if not pending_replies:
        st.info("ℹ️ No pending replies to send")
        return
    
    success_count = error_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, reply_item in enumerate(pending_replies):
        email = reply_item['email']
        draft = reply_item['draft']
        recipient_match = re.search(r'<(.+?)>', str(email.get('from','')))
        recipient = recipient_match.group(1) if recipient_match else str(email.get('from',''))
        
        status_text.text(f"📧 Sending email {i+1} of {len(pending_replies)} to {recipient[:30]}...")
        success, message = send_email(email_user, email_pass, recipient, str(email.get('subject','')), draft)
        
        if success:
            success_count += 1
            for j, orig in enumerate(st.session_state.generated_replies):
                if orig['email']==email:
                    st.session_state.generated_replies[j]['status']='sent'
                    break
        else:
            error_count += 1
        
        progress_bar.progress((i+1)/len(pending_replies))
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    if success_count>0:
        st.success(f"✅ Successfully sent {success_count} emails!")
    if error_count>0:
        st.error(f"❌ Failed to send {error_count} emails.")
    st.rerun()

def save_all_drafts():
    if not st.session_state.generated_replies:
        st.info("ℹ️ No replies to save")
        return
    
    existing = load_data('drafts.json') if os.path.exists('drafts.json') else []
    for item in st.session_state.generated_replies:
        existing.append({
            'subject':str(item['email'].get('subject','')),
            'from':str(item['email'].get('from','')),
            'reply':item['draft'],
            'saved_at':datetime.now().isoformat(),
            'original_email':str(item['email'].get('customer',''))[:500]
        })
    
    if save_data(existing,'drafts.json'):
        st.success(f"💾 Saved {len(st.session_state.generated_replies)} drafts!")
    else:
        st.error("❌ Failed to save drafts")

def display_analytics():
    emails = load_data(CSV_FILE)
    email_user = st.session_state.get('email_user', '')
    
    if not emails:
        st.info("📭 No email data available. Load emails first to see analytics.")
        return
    
    # Load KB from Google Drive for analytics
    kb = load_kb_from_drive(email_user) if email_user else {'total': len(emails), 'with_replies': 0, 'emails': []}
    
    st.subheader("📊 Email Overview")
    display_metrics(kb)
    
    st.subheader("🏷️ Common Keywords")
    all_keywords = []
    for e in emails:
        all_keywords.extend(extract_keywords(str(e.get('customer',''))))
    
    if all_keywords:
        from collections import Counter
        keyword_counts = Counter(all_keywords).most_common(10)
        cols = st.columns(min(5,len(keyword_counts)))
        for i,(kw,count) in enumerate(keyword_counts[:5]):
            with cols[i]:
                st.metric(f"#{i+1} {kw.title()}", count)
        
        if len(keyword_counts)>5:
            with st.expander("View All Keywords"):
                for kw,count in keyword_counts:
                    st.write(f"**{kw.title()}**: {count} emails")
    
    st.subheader("📈 Response Statistics")
    total_emails = len(emails)
    replied = len([e for e in emails if e.get('has_reply',False)])
    
    c1,c2,c3 = st.columns(3)
    with c1:
        rate = (replied/total_emails*100) if total_emails>0 else 0
        st.metric("Response Rate", f"{rate:.1f}%")
    with c2:
        st.metric("Total Emails", total_emails)
    with c3:
        st.metric("Pending Responses", total_emails - replied)

def system_settings():
    email_user = st.session_state.get('email_user', '')
    
    st.subheader("🔧 System Maintenance")
    c1,c2 = st.columns(2)
    with c1:
        st.write("**Local Data Files:**")
        for fn in [CSV_FILE, 'drafts.json', LOG_FILE]:
            if os.path.exists(fn):
                size_mb = os.path.getsize(fn)/(1024*1024)
                st.write(f"✅ {fn}: {size_mb:.2f} MB")
            else:
                st.write(f"❌ {fn}: Not found")
        
        st.write("**Google Drive Knowledge Base:**")
        if email_user:
            kb_status = load_kb_from_drive(email_user)
            if kb_status.get('total', 0) > 0:
                st.write(f"✅ KB on Drive: {kb_status.get('total', 0)} emails")
                if kb_status.get('last_backup'):
                    st.write(f"🕒 Last updated: {kb_status['last_backup'][:19]}")
            else:
                st.write("❌ No KB found on Google Drive")
        else:
            st.write("❌ No email user configured")
    
    with c2:
        st.write("**Actions:**")
        if st.button("🗑️ Clear Local Data"):
            for fn in [CSV_FILE, 'drafts.json', CUSTOM_PROMPT_FILE]:
                if os.path.exists(fn):
                    os.remove(fn)
            # Clear session state but don't remove KB from Drive
            for key in list(st.session_state.keys()):
                if key not in ['email_user', 'email_pass']:  # Keep credentials
                    del st.session_state[key]
            st.success("✅ Local data cleared (KB remains on Google Drive)")
            st.rerun()
        
        if st.button("📊 Export Local Data"):
            export_data = {}
            for fn,key in [(CSV_FILE,'emails'),('drafts.json','drafts'),(CUSTOM_PROMPT_FILE,'custom_prompt')]:
                if os.path.exists(fn):
                    export_data[key] = load_data(fn)
            
            # Include KB from Google Drive in export
            if email_user:
                kb_data = load_kb_from_drive(email_user)
                if kb_data.get('total', 0) > 0:
                    export_data['knowledge_base'] = kb_data
            
            export_data.update({'exported_at':datetime.now().isoformat(),'version':"2.0"})
            export_json = json.dumps(export_data,indent=2,ensure_ascii=False)
            st.download_button(
                "📥 Download Data Export",
                data=export_json,
                file_name=f"email_assistant_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.divider()
    st.subheader("🎯 Custom AI Prompt Settings")
    custom_prompt, custom_enabled = load_custom_prompt()
    
    st.markdown("""
    <div class="custom-prompt-box">
        <h4>🤖 Customize AI Reply Generation</h4>
        <p>Add custom instructions to personalize how the AI generates email replies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1,col2 = st.columns([3,1])
    with col1:
        new_custom_prompt = st.text_area(
            "Additional AI Instructions:",
            value=custom_prompt,
            height=150,
            placeholder="Example: Always include our company phone number (555-123-4567). Use a friendly tone. Mention our 30-day return policy when relevant."
        )
    with col2:
        status_color = "🟢" if custom_enabled else "🔴"
        st.write(f"{status_color} {'Enabled' if custom_enabled else 'Disabled'}")
        if custom_prompt.strip():
            wc = len(custom_prompt.split())
            st.write(f"📝 Words: {wc}")
            if wc>100:
                st.warning("⚠️ Long prompts may affect performance")
    
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        if st.button("✅ Enable"):
            if new_custom_prompt.strip() and save_custom_prompt(new_custom_prompt, True):
                st.success("✅ Custom prompt enabled!")
                st.rerun()
    with c2:
        if st.button("❌ Disable"):
            if save_custom_prompt(new_custom_prompt, False):
                st.success("❌ Custom prompt disabled!")
                st.rerun()
    with c3:
        if st.button("💾 Save"):
            if save_custom_prompt(new_custom_prompt, custom_enabled):
                st.success("💾 Prompt saved!")
                st.rerun()
    with c4:
        if st.button("🗑️ Clear"):
            if save_custom_prompt("", False):
                st.success("🗑️ Prompt cleared!")
                st.rerun()
    
    st.divider()
    st.subheader("🧠 Comprehensive Knowledge Base")
    
    st.markdown("""
    <div class="custom-prompt-box">
        <h4>📊 Build Complete Email Knowledge Base</h4>
        <p>Fetch <strong>ALL</strong> emails from your account to create a comprehensive knowledge base on Google Drive.</p>
    </div>
    """, unsafe_allow_html=True)
    
    email_pass = st.session_state.get('email_pass','')
    if not email_user or not email_pass:
        st.warning("⚠️ Please configure Gmail credentials first")
    else:
        col1,col2 = st.columns(2)
        with col1:
            # Load KB status from Google Drive
            kb = load_kb_from_drive(email_user)
            if kb.get('total', 0) > 0:
                is_comp = kb.get('comprehensive',False)
                tot = kb.get('total',0)
                if is_comp:
                    st.success(f"✅ Comprehensive KB on Drive: {tot} emails")
                    stats = kb.get('statistics',{})
                    st.info(f"📊 With replies: {stats.get('with_replies',0)}")
                    dr = stats.get('date_range',{})
                    if dr.get('earliest') and dr.get('latest'):
                        st.info(f"📅 Range: {dr['earliest'][:10]} to {dr['latest'][:10]}")
                else:
                    st.info(f"📋 Standard KB on Drive: {tot} emails")
                    st.warning("⚠️ Not comprehensive")
            else:
                st.info("📭 No knowledge base found on Google Drive")
        
        with col2:
            st.warning("⚠️ **Important:** This fetches ALL emails and may take 30+ minutes for large accounts")
            if st.button("🚀 Build Comprehensive KB", type="primary"):
                if not os.path.exists(DRIVE_TOKEN_FILE):
                    st.error("❌ Google Drive not configured. Please set up Google Drive integration first.")
                else:
                    try:
                        pc = st.container()
                        with pc:
                            st.info("🚀 Starting comprehensive build...")
                            pb = st.progress(0)
                            st_text = st.empty()
                            
                            def upd(pct,msg):
                                pb.progress(pct/100)
                                st_text.text(f"🔄 {msg}")
                            
                            upd(5, "Connecting to Gmail...")
                            inbox_emails, sent_emails = comprehensive_email_fetch(email_user, email_pass, upd)
                            
                            upd(90,"Processing and saving to Google Drive...")
                            pairs = match_replies(inbox_emails, sent_emails)
                            
                            if save_data(pairs, CSV_FILE):
                                kb = create_knowledge_base(pairs, email_user, True)
                                kb['last_updated']=datetime.now().isoformat()
                                kb['build_type']='comprehensive'
                                kb['total_fetched']=len(inbox_emails)+len(sent_emails)
                                
                                pb.progress(100)
                                st_text.text("✅ Complete!")
                                st.success("🎉 Comprehensive KB Built and Saved to Google Drive!")
                                
                                c1,c2,c3 = st.columns(3)
                                with c1: st.metric("📧 Total Emails", len(inbox_emails)+len(sent_emails))
                                with c2: st.metric("🔗 Conversations", len(pairs))
                                with c3: st.metric("✅ With Replies", kb.get('with_replies',0))
                                
                                if kb.get('backup_status')=='success':
                                    st.success("☁️ Successfully saved to Google Drive!")
                                else:
                                    st.warning(f"⚠️ Google Drive save issue: {kb.get('backup_error', 'Unknown')}")
                                
                                time.sleep(2)
                                pc.empty()
                                st.rerun()
                            else:
                                st.error("❌ Failed to save email data")
                    except Exception as e:
                        st.error(f"❌ Build failed: {str(e)}")
            
            if st.button("📊 Estimate Email Count"):
                try:
                    with st.spinner("📊 Counting emails..."):
                        mail = login_gmail(email_user, email_pass)
                        mail.select('INBOX')
                        _, inbox_msgs = mail.search(None,'ALL')
                        inbox_count = len(inbox_msgs[0].split()) if inbox_msgs[0] else 0
                        
                        mail.select('"[Gmail]/Sent Mail"')
                        _, sent_msgs = mail.search(None,'ALL')
                        sent_count = len(sent_msgs[0].split()) if sent_msgs[0] else 0
                        mail.logout()
                        
                        total_count = inbox_count+sent_count
                        est_minutes = (total_count*2)/60
                        
                        st.success("📊 Email Count Estimation:")
                        c1,c2,c3 = st.columns(3)
                        with c1: st.metric("📧 Inbox", inbox_count)
                        with c2: st.metric("📤 Sent", sent_count)
                        with c3: st.metric("📊 Total", total_count)
                        
                        time_str = f"{est_minutes:.0f} minutes" if est_minutes<60 else f"{est_minutes/60:.1f} hours"
                        st.info(f"⏱️ **Estimated Processing Time:** {time_str}")
                        
                        if total_count>10000:
                            st.warning("⚠️ **Large Account (10k+ emails)** - May take 2+ hours")
                        elif total_count>5000:
                            st.info("📈 **Medium Account (5k+ emails)** - Should complete in 1-2 hours")
                        else:
                            st.success("✅ **Small Account (<5k emails)** - Should complete quickly")
                except Exception as e:
                    st.error(f"❌ Error counting emails: {str(e)}")

    # Google Drive Integration Section
    st.divider()
    st.subheader("☁️ Google Drive Integration")
    if not GOOGLE_DRIVE_AVAILABLE:
        st.error("❌ Google Drive libraries not installed")
        st.code("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    else:
        drive_tab1, drive_tab2, drive_tab3 = st.tabs(["📤 Setup", "📊 Status", "🔄 Backup/Restore"])
        
        with drive_tab1:
            st.info("""
            **Setup Instructions:**
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select existing one
            3. Enable Google Drive API
            4. Create credentials (OAuth 2.0 Client ID)
            5. Download the credentials JSON file
            6. Upload it below
            """)
            uploaded_file = st.file_uploader(
                "Upload Google API Credentials JSON",
                type=['json'],
                help="Download from Google Cloud Console"
            )
            if uploaded_file is not None:
                try:
                    credentials_data = json.load(uploaded_file)
                    with open(DRIVE_CREDS_FILE, 'w') as f:
                        json.dump(credentials_data, f, indent=2)
                    st.success("✅ Credentials file saved!")
                except Exception as e:
                    st.error(f"❌ Invalid credentials file: {str(e)}")
            
            # Use the new Streamlit-compatible OAuth flow
            if st.button("🔐 Authorize Google Drive Access", type="primary"):
                streamlit_google_oauth()

        with drive_tab2:
            st.write("**Google Drive Status:**")
            if os.path.exists(DRIVE_TOKEN_FILE):
                st.success("✅ Google Drive configured")
                try:
                    creds, _ = setup_google_drive_auth()
                    if creds:
                        service = build('drive', 'v3', credentials=creds)
                        about = service.about().get(fields="user").execute()
                        user_email = about.get('user', {}).get('emailAddress', 'Unknown')
                        st.info(f"📧 Connected as: {user_email}")
                        
                        # Check knowledge bases folder status
                        folder_id, folder_message = ensure_knowledge_bases_folder()
                        if folder_id:
                            st.success("📁 Knowledge bases folder: ✅ Ready")
                            
                            # Count KB files in folder
                            try:
                                results = service.files().list(
                                    q=f"parents='{folder_id}' and trashed=false and name contains 'knowledge_base.json'",
                                    fields="files(id,name)"
                                ).execute()
                                kb_files = results.get('files', [])
                                st.info(f"📊 Knowledge base files in folder: {len(kb_files)}")
                            except:
                                pass
                        else:
                            st.warning(f"📁 Knowledge bases folder: ❌ {folder_message}")
                        
                        if st.button("🧪 Test Connection"):
                            # Test by trying to load KB
                            test_kb = load_kb_from_drive(email_user) if email_user else {}
                            st.success(f"✅ Connected to Google Drive as: {user_email}")
                            if test_kb.get('total', 0) > 0:
                                st.info(f"📊 Knowledge Base: {test_kb.get('total', 0)} emails found")
                            else:
                                st.info("📊 No knowledge base found for current user")
                except Exception as e:
                    st.warning(f"⚠️ Connection issue: {str(e)}")
                
                # Show KB status from Google Drive
                if email_user:
                    kb_status = load_kb_from_drive(email_user)
                    if kb_status.get('total', 0) > 0:
                        backup_time = kb_status.get('last_backup', kb_status.get('created_at', 'Unknown'))
                        st.success(f"✅ KB loaded from Drive: {backup_time[:19] if len(backup_time)>19 else backup_time}")
                        
                        # Show file organization info
                        st.info("📁 Files are organized in 'knowledge bases' folder")
                        drive_filename = get_drive_filename(email_user, 'knowledge_base.json')
                        st.code(f"File location: /knowledge bases/{drive_filename}")
                    else:
                        st.warning("⚠️ No knowledge base found on Google Drive")
                
                if st.button("🗑️ Reset Authorization"):
                    try:
                        if os.path.exists(DRIVE_TOKEN_FILE): os.remove(DRIVE_TOKEN_FILE)
                        if os.path.exists(DRIVE_CREDS_FILE): os.remove(DRIVE_CREDS_FILE)
                        st.success("✅ Authorization reset!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Reset failed: {str(e)}")
            else:
                st.warning("⚠️ Google Drive not configured")
                st.info("Use the Setup tab to configure Google Drive integration")

        with drive_tab3:
            st.write("**Backup & Restore Operations:**")
            if not email_user:
                st.warning("⚠️ Please configure Gmail credentials first")
            elif not os.path.exists(DRIVE_TOKEN_FILE):
                st.warning("⚠️ Please setup Google Drive integration first")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Manual Backup:**")
                    if st.button("☁️ Backup to Google Drive", type="primary"):
                        # Load current KB from Drive and display info
                        current_kb = load_kb_from_drive(email_user)
                        if current_kb.get('total', 0) == 0:
                            st.error("❌ No knowledge base found to backup")
                        else:
                            with st.spinner("☁️ Backing up to Google Drive..."):
                                success, message = save_kb_to_drive(current_kb, email_user)
                                if success:
                                    st.success(f"✅ {message}")
                                    st.info("📁 File saved in organized 'knowledge bases' folder")
                                else:
                                    st.error(f"❌ {message}")
                
                with col2:
                    st.write("**Restore from Google Drive:**")
                    if st.button("📥 Refresh from Google Drive"):
                        with st.spinner("📥 Loading from Google Drive..."):
                            try:
                                kb_data = load_kb_from_drive(email_user)
                                if kb_data.get('total', 0) > 0:
                                    st.success("✅ Knowledge base loaded from Google Drive!")
                                    st.info(f"📊 Loaded KB contains {kb_data.get('total',0)} emails")
                                    st.info("📁 Loaded from organized 'knowledge bases' folder")
                                    # Update session state
                                    st.session_state['current_kb'] = kb_data
                                    st.rerun()
                                else:
                                    st.error(f"❌ No knowledge base found on Google Drive for {email_user}")
                            except Exception as e:
                                st.error(f"❌ Load failed: {str(e)}")
                
                # Show folder structure info
                st.divider()
                st.subheader("📁 Google Drive Organization")
                st.info("""
                **File Structure:**
                ```
                📁 Google Drive (Root)
                └── 📁 knowledge bases/
                    ├── 📄 user1_knowledge_base.json
                    ├── 📄 user2_knowledge_base.json
                    └── 📄 ...
                ```
                
                **Benefits:**
                - ✅ Organized storage in dedicated folder
                - ✅ No duplicate files (automatic overwrite)
                - ✅ Easy to find and manage
                - ✅ Consistent naming convention
                """)
                
                if email_user:
                    expected_filename = get_drive_filename(email_user, 'knowledge_base.json')
                    st.code(f"Your KB file: /knowledge bases/{expected_filename}")

def main():
    load_custom_css()
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🤖 AI Email Assistant</h1>
        <p style="font-size: 1.2rem; color: #666;">Streamline customer support with AI-powered email management via Google Drive</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        with st.expander("📧 Email Settings", expanded=True):
            email_user = st.text_input("📧 Gmail Address", placeholder="your-email@gmail.com")
            email_pass = st.text_input("🔑 Gmail App Password", type='password')
            if email_user and email_pass:
                st.session_state.update({'email_user':email_user,'email_pass':email_pass})
        
        with st.expander("🤖 AI Settings"):
            try:
                if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your-api-key"):
                    st.success("✅ OpenAI API key configured")
                else:
                    st.error("⚠️ OpenAI API key not configured!")
            except:
                st.error("⚠️ OpenAI API key not found in secrets!")
        
        st.header("📊 System Status")
        csv_exists = os.path.exists(CSV_FILE)
        drive_configured = os.path.exists(DRIVE_TOKEN_FILE)
        
        st.write("**File Status:**")
        st.write(f"📁 Emails CSV: {'✅' if csv_exists else '❌'}")
        st.write(f"☁️ Google Drive: {'✅' if drive_configured else '❌'}")
        
        # Show KB status from Google Drive
        email_user = st.session_state.get('email_user', '')
        if email_user and drive_configured:
            kb_status = load_kb_from_drive(email_user)
            if kb_status.get('total', 0) > 0:
                st.write(f"🧠 KB on Drive: ✅ ({kb_status.get('total', 0)} emails)")
                if kb_status.get('last_updated'):
                    try:
                        last = datetime.fromisoformat(kb_status['last_updated'])
                        hours_ago = (datetime.now()-last).total_seconds()//3600
                        st.write(f"🕒 Updated: {hours_ago:.0f}h ago")
                    except:
                        st.write(f"🕒 Updated: {kb_status['last_updated'][:19]}")
            else:
                st.write("🧠 KB on Drive: ❌")
        else:
            st.write("🧠 KB on Drive: ❌")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search & Load","✍️ Reply Management","📊 Analytics","⚙️ Settings"])
    
    with tab1:
        st.header("🔍 Email Search & Loading")
        
        # Check if Google Drive is configured first
        if not os.path.exists(DRIVE_TOKEN_FILE):
            st.error("❌ Google Drive integration required! Please configure it in the Settings tab first.")
            st.info("👉 Go to Settings > Google Drive Integration > Setup to get started")
            return
        
        c1,c2,c3,c4 = st.columns(4)
        with c1: search_type = st.selectbox("🔍 Search Type:", ["all","sender","subject","body"])
        with c2: search_value = st.text_input("🔎 Search Term:", placeholder="Enter term...") if search_type!="all" else ""
        with c3: date_from = st.date_input("📅 From Date:", value=None)
        with c4: date_to = st.date_input("📅 To Date:", value=None)
        
        with st.expander("⚙️ Advanced Options"):
            ca,cb = st.columns(2)
            with ca: email_limit = st.slider("📧 Max Emails:",50,2000,200,step=50)
            with cb: folder_option = st.selectbox("📁 Folder:",["inbox","sent","both"])
        
        ca,cb = st.columns(2)
        with ca:
            if st.button("🔍 Standard Search & Load",type="primary",use_container_width=True):
                eu = st.session_state.get('email_user','')
                ep = st.session_state.get('email_pass','')
                if not eu or not ep:
                    st.error("❌ Please configure Gmail credentials")
                else:
                    try:
                        # Test OpenAI API key
                        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your-api-key"):
                            st.error("❌ OpenAI API key not properly configured")
                        else:
                            search_and_load_emails(eu,ep,search_type,search_value,date_from,date_to,email_limit,"standard")
                    except:
                        st.error("❌ OpenAI API key not found in Streamlit secrets")
        
        with cb:
            if st.button("🚀 Comprehensive Build (ALL Emails)",use_container_width=True):
                eu = st.session_state.get('email_user','')
                ep = st.session_state.get('email_pass','')
                if not eu or not ep:
                    st.error("❌ Please configure Gmail credentials")
                else:
                    search_and_load_emails(eu,ep,search_type,search_value,date_from,date_to,email_limit,"comprehensive")

    with tab2:
        st.header("✍️ Email Reply Management")
        
        # Check if Google Drive is configured first
        if not os.path.exists(DRIVE_TOKEN_FILE):
            st.error("❌ Google Drive integration required! Please configure it in the Settings tab first.")
            st.info("👉 Go to Settings > Google Drive Integration > Setup to get started")
            return
        
        reply_management_section(
            st.session_state.get('email_user',''),
            st.session_state.get('email_pass','')
        )

    with tab3:
        st.header("📊 Email Analytics")
        display_analytics()

    with tab4:
        st.header("⚙️ System Settings")
        system_settings()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        log(f"Critical error: {str(e)}", "error")