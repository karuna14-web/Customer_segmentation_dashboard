# Customer Segmentation Dashboard - Recovery Guide

## Starting the Project After a Google Colab Runtime Reset

### 1. Open the Google Colab notebook
Open the notebook used to run the Customer Segmentation Dashboard.

### 2. Mount Google Drive
Run the Google Drive mounting cell if Drive is not already mounted.

### 3. Run the Automatic Recovery Cell
Run the saved automatic recovery cell.

It will:
- Restore Google OAuth Client ID and Client Secret securely if missing.
- Generate a Flask Secret Key if missing.
- Install Cloudflare Tunnel if it is not installed.
- Start Cloudflare Tunnel.
- Detect the new temporary Cloudflare URL.
- Set the PUBLIC_BASE_URL environment variable.
- Start the Flask application.
- Verify the Flask server and public tunnel.
- Display the new URLs needed for Google Cloud OAuth configuration.

### 4. Update Google Cloud OAuth settings
Open Google Cloud Console:
https://console.cloud.google.com/apis/credentials

Open the OAuth Client for Customer Segmentation Dashboard.

Update the following:

Authorized JavaScript origins:
The new Cloudflare URL printed by the recovery cell.

Authorized redirect URI:
The new Cloudflare URL followed by /auth/google/callback.

Save the changes.

### 5. Test Google Login
Open the new public dashboard URL.

Click Continue with Google and confirm that you are redirected to the dashboard.

### Important
- The Cloudflare Quick Tunnel URL changes after a runtime reset.
- Update Google Cloud OAuth settings whenever the URL changes.
- Keep the Colab runtime running while using the dashboard.
- Do not share Google Client Secret or Flask Secret Key.
- Google OAuth credentials should be entered securely through the notebook.
- This recovery process uses a temporary Cloudflare Quick Tunnel.
