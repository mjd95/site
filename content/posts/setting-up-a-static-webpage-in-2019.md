---
title: "Setting up a static web page in 2019"
date: 2019-09-23T07:26:00+01:00
---
I was recently setting up a website for a small [travel blog](https://funtastictravels.business).  We wanted a reasonable looking static site, content supplied via markdown files, hosted on a custom domain and supporting TLS.  I'm not sure what the quickest way to do this is in 2019, but it probably involves Netlify.  I went down a Hugo / Google Domains / App Engine route, and it was still pretty slick.

First the static site generator.  There are _so many_ static site generators out there, but I'd never found one I liked.  In fact I wrote my own one in bash for the site you're currently reading:
```
rm -rf www 
mkdir -p www/posts

# generate all the post pages
pushd content/posts
for filename in *.html; do
	cat ../../includes/meta.html ../../includes/header.html "$filename" ../../includes/footer.html > ../../www/posts/"$filename"
done

# generate the body of the post aggregation page
rm ../posts.html
echo '<div class="container home">' >> ../posts.html
posts=`ls . | sort -r`
for filename in $posts; do
	lastline=`cat $filename | grep -n '</p>' | cut -d':' -f1 | head -n 1`
	snippet=`cat $filename | tail -n +2 | head -n $(($lastline-1))`
	echo "$snippet" >> ../posts.html
	echo "<a href="/posts/$filename">(read more)</a>" >> ../posts.html
done
echo '</div>' >> ../posts.html
popd

# generate top level pages
pushd content
for filename in \*.html; do
	basefilename=${filename::-5}
	mkdir -p ../www/$basefilename
	cat ../includes/meta.html ../includes/header.html "$filename" ../includes/footer.html > ../www/"$basefilename"/index.html
done
popd

mkdir -p www/css
cp css/style.css www/css/style.css
```
The less said about this the better.

This time, I tried [Hugo](https://gohugo.io/) again, and it was easier to get something I liked the look of than I remember.  I followed the [quickstart](https://gohugo.io/getting-started/quick-start/) and found a sufficiently frivolous [theme](https://themes.gohugo.io/theme/papercss-hugo-theme/), and had something which looked like what I wanted running on localhost within 5 minutes.

I got sidetracked at this point by the `hugo deploy` [command](https://gohugo.io/hosting-and-deployment/hugo-deploy/) which I could use to push my static site to a GCS bucket.  I found two issues (probably both surmountable) with a purely bucket-based site though:
  * Visiting `http://mysite.com` should show you the HTML in `http://mysite.com/index.html`, but this behaviour didn't come for free from the bucket and I wasn't sure how to configure it
  * Adding TLS support to bucket-based site seems to require a few [extra hops](https://geekflare.com/gcs-site-over-https/)

Instead I decided to push to App Engine.  App Engine is pretty flexible, but it is in particular capable of doing static bucket-backed sites with a regex-based router in front, and offers easy TLS setup.  This is pretty much exactly what I needed.

First focus on deploying a site with correct routing.  If I generate my Hugo site:
```
 > cd mysite
 > hugo
 > ls public
android-chrome-192x192.png
android-chrome-512x512.png
apple-touch-icon.png
browserconfig.xml
categories
css
favicon-16x16.png
favicon-32x32.png
favicon.ico
img
index.html
index.xml
mstile-144x144.png
mstile-150x150.png
mstile-310x150.png
mstile-310x310.png
mstile-70x70.png
posts
safari-pinned-tab.svg
sitemap.xml
site.webmanifest
tags
```
then there's only a few things I actually care about in there.  Specifically, I want the `index.html`, the `posts` directory and all its contents (an `index.html` and, for each post, a file `posts/mypost/index.html`), and finally the `css` directory.  I wrote an `app.yaml` file that would capture everything I cared about and do the expected redirects:
```
runtime: python27
api\_version: 1
threadsafe: true

handlers:
- url: /(|index.html)
  static\_files: public/index.html
  upload: public/index.html

- url: /posts/(|index.html)
  static\_files: posts/index.html
  upload: public/posts/index.html

- url: /posts/(.\*)/
  static\_files: public/posts/\1/index.html
  upload: public/posts/(.\*)/index.html

- url: /css
  static\_dir: public/css
```
To get my page running on the appspot domain, it's now just a case of `gcloud app deploy`.

Now focus on the custom domain and TLS.  I bought the domain I wanted from Google Domains and verified it using `gcloud domains verify`.  To actually use this for my static site, I simply had to run `gcloud app domain-mappings create`.  This gave me back a bunch of A/AAAA records which I added within the Google Domains UI.  (If I understand correctly, these DNS records point at the app engine load balancer, and since I have created a mapping for my URL to my app engine instance the load balancer knows to reverse proxy incoming requests to my app engine instance.)  The nicest part about all of this is that the TLS certificate generation was automatic at this point, so after a short delay for the DNS propagation and certificate generation I had my web page running with TLS on my own domain like I wanted.

I've used app engine before, but I was impressed how easy this integrates with custom domains and TLS.  The whole process was well under an hour, and would have been even less if I hadn't spent some time playing around with the direct-from-bucket idea.

I was also pleased enough with Hugo that I decided to retire my bash static site generator and port my personal web page.  Maybe that means I'm more likely to write stuff now :)
