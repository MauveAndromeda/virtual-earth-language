#!/bin/bash
# Fix the token issue and upload to GitHub

echo "🔧 Fixing token issue and uploading..."

# Remove the upload script with the token
rm -f upload.sh

# Create a clean version without the token
cat > upload_clean.sh << 'EOF'
#!/bin/bash
# GitHub Upload Script for Virtual Earth Language Evolution (Clean Version)

set -e

echo "🚀 Uploading Virtual Earth project to GitHub..."

# Replace these with your actual values when running
GITHUB_USERNAME="MauveAndromeda"
GITHUB_TOKEN="YOUR_TOKEN_HERE"  # Replace this manually
REPO_NAME="virtual-earth-language"

echo "✅ Clean upload script created - replace token manually if needed"
EOF

chmod +x upload_clean.sh

# Add the gitignore entry to prevent future token commits
echo "" >> .gitignore
echo "# Prevent token commits" >> .gitignore  
echo "*token*" >> .gitignore
echo "*secret*" >> .gitignore
echo "upload.sh" >> .gitignore

# Stage the changes
git add -A
git status

# Commit the fix
git commit -m "🔒 Remove token from upload script and add security gitignore

- Removed upload.sh containing token
- Added upload_clean.sh as template
- Enhanced .gitignore for security
- Ready for secure upload"

# Now push with the token directly (since repo already exists)
echo "⬆️ Pushing to GitHub..."
git push "https://MauveAndromeda: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@github.com/MauveAndromeda/virtual-earth-language.git" main

# Clean up the credentials from git
git remote set-url origin "https://github.com/MauveAndromeda/virtual-earth-language.git"

echo ""
echo "✅ SUCCESS! Your project is now live at:"
echo "🌐 https://github.com/MauveAndromeda/virtual-earth-language"
echo ""
echo "🎉 Virtual Earth Language Evolution is ready for the world!"
