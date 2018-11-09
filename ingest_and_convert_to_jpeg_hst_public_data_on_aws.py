print('[INFO]')
from astroquery.mast import Observations
from glob import glob
from pyql.file_system.make_fits_file_dict import make_fits_file_dict
from pyql.ingest.make_jpeg import make_jpeg
from sklearn.externals import joblib
from tqdm import tqdm

import boto3
import numpy as np
import os

print('[INFO]')
savedir = os.environ['HOME'] + '/wfc3f/'
jpeg_dir = os.environ['HOME'] + '/JPEG/'

print('[INFO]')
if not os.path.exists(savedir): os.mkdir(savedir)
if not os.path.exists(jpeg_dir): os.mkdir(jpeg_dir)

print('[INFO]')
WFC3_UVIS_Filters = ['F200LP', 'F300X', 'F350LP', 'F475X', 'F600LP', 'F850LP', 'F218W', 'F225W', 'F275W', 
    'F336W', 'F390W', 'F438W', 'F475W', 'F555W', 'F606W', 'F625W', 'F775W', 'F814W', 'F390M', 'F410M', 
    'F467M', 'F547M', 'F621M', 'F689M', 'F763M', 'F845M', 'F280N', 'F343N', 'F373N', 'F395N', 'F469N', 
    'F487N', 'F502N', 'F631N', 'F645N', 'F656N', 'F657N', 'F658N', 'F665N', 'F673N', 'F680N', 'F953N', 
    'F232N', 'F243N', 'F378N', 'F387N', 'F422M', 'F436N', 'F437N', 'F492N', 'F508N', 'F575N', 'F619N', 
    'F634N', 'F672N', 'F674N'
]

print('[INFO]')
WFC3IR_Filters = ['F105W', 'F110W', 'F125W', 'F140W', 'F160W', 'F098M', 'F127M', 'F139M', 'F153M', 
                    'F126N', 'F128N', 'F130N', 'F132N', 'F164N', 'F167N'
                 ]


WFC3IR_Filters = ['F105W', 'F110W', 'F125W', 'F140W', 'F160W', 'F098M', 'F127M', 'F139M', 'F153M', 
                    'F126N', 'F128N', 'F130N', 'F132N', 'F164N', 'F167N'
                 ]

print('[INFO]')
WFC3IR_Grisms = ['G102', 'G141']
WFC3UVIS_Grisms = ['G280']


save_files = ['obsTables_dict_all_WFC3IR_Filiters.joblib.save', 
              'products_dict_list_WFC3IR_Filiters.joblib.save', 
              'filtered_dict_list_WFC3IR_Filiters.joblib.save', 
              's3_urls_dict_list_WFC3IR_Filiters.joblib.save']

print('[INFO]')
files_in_dir = glob('*')
all_saves_exist = np.all([sname in files_in_dir for sname in save_files])

# Enable 'S3 mode' for module which will return S3-like URLs for FITs files
# e.g. s3://stpubdata/hst/public/icde/icde43l0q/icde43l0q_drz.fits
print('[INFO]')
Observations.enable_s3_hst_dataset()

print('[INFO]')
s3 = boto3.resource('s3')

# Create an authenticated S3 session. Note, download within US-East is free
# e.g. to a node on EC2.
print('[INFO]')
s3_client = boto3.client('s3',
                         aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                         aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

print('[INFO]')
hst_bucket = s3.Bucket('stpubdata')

if all_saves_exist: # Load them
    print('[INFO]')
    obsTable_dict = joblib.load('obsTables_dict_all_WFC3IR_Filiters.joblib.save')
    products_dict = joblib.load('products_dict_list_WFC3IR_Filiters.joblib.save')
    filtered_dict = joblib.load('filtered_dict_list_WFC3IR_Filiters.joblib.save')
    s3_urls_dict = joblib.load('s3_urls_dict_list_WFC3IR_Filiters.joblib.save')
else: # Create them
    print('[INFO]')
    print('[INFO]')
    obsTable_dict = {ir_filt:Observations.query_criteria(obs_collection='HST', 
                                                            instrument_name='WFC3/IR', 
                                                            filters=ir_filt) 
                                                            for ir_filt in tqdm(WFC3IR_Filters)}
    print('[INFO]')
    joblib.dump(obsTable_dict, 'obsTables_dict_all_WFC3IR_Filiters.joblib.save')
    
    # products_dict_full = {ir_filt:Observations.get_product_list(obsTable_dict[ir_filt]) for ir_filt in tqdm(WFC3IR_Filters)}
    print('[INFO]')
    chunk_size = 1000
    product_dict_full = {}
    for ir_filt in tqdm(WFC3IR_Filters):
        n_obs = len(obsTable_dict[ir_filt])
        n_chunks = n_obs // chunk_size +1
        product_dict_full[ir_filt] = []
        for k in tqdm(range(n_chunks)):
            try:
                product_dict_full[ir_filt].append(Observations.get_product_list(obsTable_dict[ir_filt][k*chunk_size:(k+1)*chunk_size]))
            except Exception as e:
                print(str(e))
    
    print('[INFO]')
    joblib.dump(product_dict_full, 'products_dict_full_listed_WFC3IR_Filiters.joblib.save')
    
    # Select only FLT files
    # mrp = minimum recommended products
    # filtered_dict = {ir_filt:Observations.filter_products(products_dict[ir_filt],
    #                         mrp_only=False, productSubGroupDescription='FLT',
    #                         dataproduct_type='image')
    #                         for ir_filt in tqdm(WFC3IR_Filters)}
    print('[INFO]')
    filtered_dict = {}
    for ir_filt in tqdm(WFC3IR_Filters):
        filtered_dict[ir_filt] = []
        for product_tbl in product_dict_full[ir_filt]:
            filtered_dict[ir_filt].append(Observations.filter_products(product_tbl, 
                                            mrp_only=False, productSubGroupDescription='FLT', 
                                            dataproduct_type='image'))
    
    print('[INFO]')
    joblib.dump(filtered_dict, 'filtered_dict_list_WFC3IR_Filiters.joblib.save')
    
    # Grab the S3 URLs for each of the observations
    # s3_urls_dict = {ir_filt:Observations.get_hst_s3_uris(filtered_dict[ir_filt]) for ir_filt in tqdm(WFC3IR_Filters)}
    print('[INFO]')
    s3_urls_dict = {}
    for ir_filt in tqdm(WFC3IR_Filters):
        s3_urls_dict[ir_filt] = []
        for kf, filtered_tbl in tqdm(enumerate(filtered_dict[ir_filt]), total=len(filtered_dict[ir_filt])):
            for kt, tbl_now in tqdm(enumerate(filtered_tbl), total=len(filtered_tbl)):
                try:
                    s3_urls_dict[ir_filt].append(Observations.get_hst_s3_uri(tbl_now))
                except Exception as e:
                    s3_urls_dict[ir_filt].append([kf, kt, str(e)])
    
    print('[INFO]')
    joblib.dump(s3_urls_dict, 's3_urls_dict_list_WFC3IR_Filiters.joblib.save')

print('[INFO]')
# existing_JPEGs_filenames = glob(os.path.abspath('wfc3f/JPEG/*'))
# existing_JPEGs = [fname.split('-')[-1] for fname in existing_JPEGs_filenames]
print('[INFO]')
bucket_name = 'wfc3-ir-image-jpegs'
my_bucket = s3.Bucket(bucket_name)

print('[INFO]')
bucket_list = s3_client.list_objects(Bucket = bucket_name)
existing_JPEGs = [blist_content['Key'].split('/')[-1].split('-')[-1] for blist_content in bucket_list['Contents'] if 'JPEG' in blist_content['Key']]

# s3_urls_dict_one = {'F105W':s3_urls_dict['F105W'].copy()}
print('[INFO]')
for ir_filt, s3_urls_now in tqdm(s3_urls_dict_one.items(), total=len(s3_urls_dict)):
    for url in tqdm(s3_urls_now[:1], total=len(s3_urls_now)):#[:nUrls]
        # Extract the S3 key from the S3 URL
        fits_s3_key = url.replace("s3://stpubdata/", "")
        root = url.split('/')[-1]

        fits_filename = savedir + ir_filt + '/' + root
        
        if not os.path.exists(savedir): os.mkdir(savedir)
        if not os.path.exists(savedir + ir_filt): os.mkdir(savedir + ir_filt)
        # if not os.path.exists(fits_filename): os.mkdir(fits_filename)
        
        if root.replace('.fits','.jpg') not in existing_JPEGs:
            my_bucket.download_file(fits_s3_key, fits_filename, ExtraArgs={"RequestPayer": "requester"})
            
            file_dict = make_fits_file_dict(fits_filename)
            make_jpeg(file_dict)
            os.remove(fits_filename)
            
            '''
            FINDME: 
                MOVE ALL FILES TO THEIR ANOMALY SUBDIRECTORIES INSIDE THE AWS INSTANCE; THEN SYNC TO AWS S3
                THIS REQUIRES HAVING THE ANOMALY CLASS DICTIONARY ON THE AWS INSTANCE AND LOADED HERE!
            '''
            
            jpeg_filename = os.environ['HOME'] + '/wfc3f/JPEG/' + ir_filt + '-' + root.replace('.fits','.jpg')
            jpeg_key = 'JPEG/'+jpeg_filename.split('/')[-1]
            with open(jpeg_filename, 'rb') as data:
                # s3.Bucket(bucket_name).put_object(Key='test.jpg', Body=data)
                s3_client.put_object(Bucket=bucket_name, Body=data, Key=jpeg_key)
            
            os.remove(jpeg_filename)

print('[INFO]')