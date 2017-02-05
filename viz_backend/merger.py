__author__ = 'daksh'

import cv2
import numpy
from time import time
import os
from PIL import Image
import pytesseract

f = open("discrepency.csv", "w")
f.close()


tuples=[('LEFFE BLONDE', '6 Pk 11.2 Oz Glass'),
('BROOKLYN OKTOBERFEST', '6 Pk 12 Oz Glass'),
('MAGIC HAT', '6 Pk 12 Oz Glass'),
('BUD LIGHT PLATINUM', '6 Pk 12 Oz Glass'),
('SIERRA NEVADA TORPEDO', '6 Pk 12 Oz Glass'),
('SIERRA NEVADA PALE ALE', '6 Pk 12 Oz Glass'),
('BECKS', '6 Pk 12 Oz Glass'),
('BUDWEISER', '6 Pk 12 Oz Glass'),
('CORONA LIGHT', '6 Pk 12 Oz Glass'),
('CORONA EXTRA', '6 Pk 12 Oz Glass'),
('STELLA ARTOIS', '6 Pk 11.2 Oz Glass'),
('BROOKLYN INDIA PALE', '6 Pk 12 Oz Glass'),
('HOEGAARDEN', '6 Pk 11.2 Oz Glass'),
('MILLER LITE', '12 Pk 12 Oz Can'),
('BLUE POINT TOASTED', '6 Pk 12 Oz Glass'),
('MILLER LITE', '12 Pk 12 Oz Glass'),
('COORS LIGHT', '6 Pk 12 Oz Glass'),
('BUD LIGHT', '6 Pk 12 Oz Glass'),
('MILLER LITE', '6 Pk 12 Oz Glass'),
('BROOKLYN PILSNER', '6 Pk 12 Oz Glass')]


nameDict={}
for t in tuples:
    if t[0] not in nameDict.keys():
        nameDict[t[0]]=[]
    nameDict[t[0]]+=[t[1]]

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
def levenshteinDistance(str1, str2):
    m = len(str1)
    n = len(str2)
    lensum = float(m + n)
    d = []           
    for i in range(m+1):
        d.append([i])        
    del d[0][0]    
    for j in range(n+1):
        d[0].append(j)       
    for j in range(1,n+1):
        for i in range(1,m+1):
            if str1[i-1] == str2[j-1]:
                d[i].insert(j,d[i-1][j-1])           
            else:
                minimum = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+2)         
                d[i].insert(j, minimum)
    ldist = d[-1][-1]
    ratio = (lensum - ldist)/lensum
    return {'distance':ldist, 'ratio':ratio}


#~ def checkAssociations(productLabel,ocrResult):
	#~ productLabel = productLabel.split('_').join(' ').lower()
	#~ ocrResult = ocrResult.split('\n')[0]
	#~ ocrResult = ocrResult.decode('unicode_escape').encode('ascii','ignore').lower()
	#~ m = (levenshteinDistance(tuples[0][0].lower(),ocrResult)['distance'])/len(tuples[0][0].lower())
	#~ m_label = tuples[0][0].lower()
	#~ for i in range(1,len(tuples):
		#~ dist = (levenshteinDistance(tuples[i][0].lower(),ocrResult)['distance'])/len(tuples[i][0].lower())
		#~ if dist < m:
			#~ m = dist
			#~ m_label = tuples[i][0].lower()
			#~ 
	#~ if productLabel.contains(['1'])
		#~ productLabel = productLabel[:-1]
	#~ elif productLabel.
	#~ if m_label is not productLabel:
		
def checkAssociations(week, productLabel,ocrResult):		
	currImage=ocrResult.split('\n')
	currImage2=[]
	for i in currImage:
		if i!='':
			try:
				currImage2+=[i.decode('unicode_escape').encode('ascii','ignore')]
			except:
				continue
	currImage=currImage2
	#currImage[0]=currImage[0][1:]    
	#print currImage
	maxScore=0
	bestMatch=''
	for brand in nameDict.keys():
		if len(currImage)==0:
			continue
		length= max(len(brand), len(currImage[0]))*1.0
		score=1-levenshtein(brand, currImage[0])/length
		if score>maxScore:
			bestMatch=brand
			maxScore=score
	#print bestMatch, currImage[0]
	bestBrandMatch=bestMatch
	maxScore=0
	bestMatch=''
	if len(currImage)<2 or bestBrandMatch=='':
		return 1
	for subbrand in nameDict[bestBrandMatch]:
		if len(currImage)==0:
			continue
		length= max(len(subbrand), len(currImage[1]))*1.0
		score=1-levenshtein(subbrand, currImage[1])/length
		if score>maxScore:
			bestMatch=subbrand
			maxScore=score
		if len(currImage)==0:
			continue
	bestBrandMatch = bestBrandMatch.lower()
	bestMatch = bestMatch.lower()
	#print bestBrandMatch, bestMatch, '|', currImage[0], currImage[1]
	#print bestBrandMatch, bestMatch
	productLabel = " ".join(productLabel.split('_'))
	
	if '1' in productLabel :
		productLabel = productLabel[:-1]
	if 'can' in productLabel and 'can' in bestMatch:
		productLabel=productLabel[:-4]
		if productLabel == bestBrandMatch:
			return 1
	if '6' in productLabel and '6' in bestMatch:
		productLabel=productLabel[:-2]
		if productLabel == bestBrandMatch:
			return 1
	if productLabel == bestBrandMatch:
		return 1
	else :
		f = open("discrepency.csv", "a")
		f.write(week+','+productLabel+','+bestBrandMatch+'_'+bestMatch+'\n')
		f.close()

data_root = '/home/prerit/workspace/ABinBev/mergedCode/data/images/dataset/'
files = os.listdir(data_root)
haar_results = 'haar_classifier/results'
for i in os.listdir(haar_results):
    occurences_detected = i+'/coordinates.txt'
    f = open(haar_results + '/' + occurences_detected,'r')
    print occurences_detected
    lines = f.readlines()
    f.close()
    # print lines
    # break
    detections = []
    for line in lines:
        detection = line.split(',')
        det = []
        # print detection
        for c in range(1,5):
            # print c,detection[c]
            if c<4:
                det.append(int(detection[c]))
            else:
                det.append(int(detection[c][:-1]))
        detections.append(det)
    detections = sorted(detections, key=lambda x: [x[1],x[0]])
    # print detections
    # break
    imageName = occurences_detected.split('/')[0] + '.jpg'

    ### Extract Product boxes
    img = cv2.imread(data_root + imageName)
    img2 = img[:,:,2]
    img2 = img2 - cv2.erode(img2, None)

    products = {}
    beerBoxes = open('results/beer_boxes.txt','r')
    '''for j in os.listdir(data_root+'templates/'):
        template = cv2.imread(data_root+'templates/'+j)[:,:,2]
        template = template - cv2.erode(template, None)

        start = time()
        ccnorm = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)
        print ccnorm.max()
        threshold = 0.74
        # loc = numpy.where(ccnorm == ccnorm.max())
        loc = numpy.where(ccnorm >= threshold)
        th, tw = template.shape[:2]

        for pt in zip(*loc[::-1]):
            if ccnorm[pt[::-1]] < threshold:
                continue
            cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th),
                    (0, 0, 255), 2)
        products[j[:-4]] = [pt[0],pt[1],tw,th]
        beerBoxes.write(j[:-4]+','+str(pt[0])+','+str(pt[1])+','+str(tw)+','+str(th)+'\n')
        '''
    lines = beerBoxes.readlines()
    for line in lines:
        detection = line.split(',')
        det = []
        # print detection
        products[detection[0]] = []
        for c in range(1,5):
            # print c,detection[c]
            if c<4:
				products[detection[0]].append(int(detection[c]))
            else:
                products[detection[0]].append(int(detection[c][:-1]))
        #~ detections.append(det)
        cv2.rectangle(img,(products[detection[0]][0],products[detection[0]][1]),
        (products[detection[0]][0]+products[detection[0]][2],products[detection[0]][1]+products[detection[0]][3]),(0, 0, 255), 4)
    beerBoxes.close()
    #~ break
    allProducts = [products[k] for k in products.keys()]
    allProducts = sorted(allProducts, key=lambda x: [x[1],x[0]])

    associations = {}
    for j in products.keys():
        [x,y,w,h] = products[j]
        xStart = x
        xEnd = x+w
        yBelowStart = y+h
        yBelowEnd = y+h+200
        flag = False
        for k in detections:
            [boxX,boxY,boxW,boxH] = k
            if boxX>=(xStart-25) and (boxX+boxW)<=(xEnd+25) and boxY>yBelowStart and (boxY+boxH)<yBelowEnd:
                associations[j] = [products[j],k]
                flag = True
                break
        if not flag:
            associations[j] = [products[j],[x, yBelowStart, w, 200]]

    print associations,len(associations.keys())
    #~ resizedimg = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
    #~ cv2.imshow('original',resizedimg)
    #~ cv2.waitKey(0)
    #~ cv2.imwrite('results/boxes.jpg',img)
    
    for j in associations.keys():
		productBox = associations[j][0]
		tagBox = associations[j][1]
		
		priceBox = img[tagBox[1]:tagBox[1]+tagBox[3],tagBox[0]:tagBox[0] + tagBox[2],:]
		
		
		new_img = Image.fromarray(priceBox)
		txt = pytesseract.image_to_string(new_img).strip()
		checkAssociations(imageName.split('_')[1][:-4],j, txt)
		
		print j,txt,"\n---------------\n"
		font = cv2.FONT_HERSHEY_SIMPLEX
		#~ cv2.putText(img,j,(tagBox[0],tagBox[1]), font, 1,(255,255,255),2,cv2.CV_AA)
		cv2.rectangle(img,(tagBox[0],tagBox[1]),(tagBox[0]+tagBox[2],tagBox[1]+tagBox[3]),(0,0,0), 4)
    cv2.imwrite('results/'+imageName, img)
    break
#f.close()
cv2.destroyAllWindows()
