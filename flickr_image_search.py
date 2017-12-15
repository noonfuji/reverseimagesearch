''' This file contains flickr API'''

from flickrapi import FlickrAPI

FLICKR_PUBLIC = 'c0ac853a5b35e99260b1f49f8e36beca'
FLICKR_SECRET = 'c480f478aaad0d9a'

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
extras='url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'
cats = flickr.photos.search(text='kitten', per_page=5, extras=extras)
photos = cats['photos']
from pprint import pprint
pprint(photos)