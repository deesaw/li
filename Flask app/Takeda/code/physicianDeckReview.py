from libraries import *

## Load vgg16 pre-trained model
vgg16 = keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
## Extracted features
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)

##To get feature vector
def get_feature_vector(img):

    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

##Cosine similarity
def calculate_similarity(vector1, vector2):
    return (1-spatial.distance.cosine(vector1, vector2))

def pre_process(image):

    # Perform morph operations, first open to remove noise, then close to combine
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

    # Find enclosing boundingbox and crop ROI
    coords = cv2.findNonZero(close)
    x,y,w,h = cv2.boundingRect(coords)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    crop = image[y:y+h, x:x+w]

    result = crop

    return result

def preprocessimage(image):
    image1=image.split('::')[0]
    print("preprocessing")
    print(image1)
    contrast_file = cv2.imread(image1)
    contrast_file = cv2.resize(contrast_file,(224,224),3)
    f1=get_feature_vector(contrast_file)
    image2=image.split('::')[1]
    print(image2)
    contrast_file = cv2.imread(image2)
    contrast_file = cv2.resize(contrast_file,(224,224),3)
    f2=get_feature_vector(contrast_file)
    similarity=calculate_similarity(f1,f2)
    return similarity

def slideSimilarity(uploaded_file):
    path = Path(os.getcwd())
    path=path.parent
    print(path)
    master = os.path.join(path,"Takeda")
    uploaded_file='uploads_f\\' + uploaded_file
    slave = os.path.join(path,uploaded_file)
    print(master)
    print(slave)
    master_files = pd.DataFrame([ file_cont for file_cont in os.listdir(master) if file_cont.split('.')[1]=='png' ],columns=['MasterName'])
    master_files['Index'] = 1
    master_files['Filepath'] =  master+'\\'+master_files['MasterName']
    master_files=master_files.head(15)
    slave_files = pd.DataFrame([ file_cont for file_cont in os.listdir(slave) if file_cont.split('.')[1]=='png' ],columns=['SlaveName'])
    slave_files['Index'] = 1
    slave_files['Filepath'] = slave+'\\'+slave_files['SlaveName']
    slave_files=slave_files.head(20)
    df_slave_master =  master_files.merge(slave_files,on='Index',how="left")
    df_slave_master['MasterPath']=df_slave_master['Filepath_x']+'::'+df_slave_master['Filepath_y']
    df_slave_master['similarity']  =df_slave_master['MasterPath'].swifter.apply(preprocessimage)
    print(df_slave_master)
    result=df_slave_master[['MasterName','SlaveName','similarity']]
    result.columns=['Brand' , 'Physician' ,'Similarity Score']
    results=pd.DataFrame()
    results=result.loc[result.groupby("Brand")["Similarity Score"].idxmax()]
    results = results[results['Similarity Score']>=0.8]
    results['Similarity Score']=results['Similarity Score'].round(decimals=2)
    #results['Comment'] =  results.apply(lambda x : "Physician slide "+x["Physician"][:-4]+" is similar to Master slide"+x["Brand"][:-4]+" and the similarity score is "+str(x['Similarity Score']),axis=1)
    results['Comment']=results.apply(lambda x : "Physician slide "+x["Physician"][:-4]+" is similar to Master slide"+x["Brand"][:-4]+" and the similarity score is "+str(x['Similarity Score']),axis=1)
    results.columns=['Brand Deck slide' , 'Physician  Deck slide' ,'Similarity Score','Comment']
    print(results)
    return results
    '''results_final_sim = results_final_sim.append(results[results['similarity'] == results['similarity'].max()])
    results_final_sim = results_final_sim[results_final_sim['similarity']>=0.8]
    results_final_sim['comment'] =  results_final_sim.apply(lambda x : "Physician slide "+x["doctor_file"][:-4]+" similar to Master slide"+x["master_file"][:-4]+" is "+str(x['similarity']),axis=1)
    print(results_final_sim)
    return results_final_sim
    '''
