
# to upload
rm -rf .git
git init
git config --local user.email i.lizmhh@gmail.com
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/hhliz/SoundscapeEcologyFeatures
git push -u --force origin master
