import os
import codecs
import csv
import time
import datetime
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

directory = '/home/vedu29/python/abinbev/results2/'

k=[i for i in os.listdir(directory)]

actualPriceDict={}
totalDiffer={}
timeCounts={}
foundPriceDict={}
actualPriceDiff={} 

with codecs.open("ABI_Price_List.csv", 'rb', 'utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    data=[i for i in reader]
for i in data[1:]:
    if i[0] not in actualPriceDict.keys():
        actualPriceDict[i[0]]={}
        actualPriceDiff[i[0]]={}
        foundPriceDict[i[0]]={}
    if i[1] not in actualPriceDict[i[0]].keys():
        actualPriceDict[i[0]][i[1]]={}
        actualPriceDiff[i[0]][i[1]]={}
        foundPriceDict[i[0]][i[1]]={}
    actualPriceDict[i[0]][i[1]][i[2]]=i[3]

for filename in k:
    firstFlag=0
    if 'txt' in filename:
        f=open(directory+'/'+filename)
        readfile=f.read()
        f.close()
        readfile=readfile.split('-------')
        for image in readfile:
            currImage=image.split('\n')
            if firstFlag:
                firstFlag=False
                currImage=currImage[1:]
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
            if len(currImage)==0:
                continue
            #print bestMatch, currImage[0]
            bestBrandMatch=bestMatch
            maxScore=0
            bestMatch=''
            if len(currImage)<2 or bestBrandMatch=='':
                continue
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
            #print bestBrandMatch, bestMatch, '|', currImage[0], currImage[1]    
            accum=[]
            if len(currImage)<3 or (len(currImage[2])==0 and len(currImage)<3):
                continue
            for letter in currImage[2]:
                if letter == '$':
                    continue
                if letter.isdigit():
                    accum+=[letter]
                if len(accum)==2:
                    accum+=['.']
            if accum:
                price=float(''.join(accum))
                try:
                    timekey=datetime.datetime.strptime(filename[18:28], '%Y-%m-%d').strftime('%d/%m/%Y')
                except:
                    continue
                if len(bestMatch)==0:
                    bestMatch=actualPriceDict[timekey][bestBrandMatch].keys()[0]
                #print timekey, price, actualPriceDict[timekey][bestBrandMatch][bestMatch],abs(float(price) -float(actualPriceDict[timekey][bestBrandMatch][bestMatch])), bestBrandMatch, bestMatch
                if timekey not in totalDiffer.keys():
                    totalDiffer[timekey]=0
                    timeCounts[timekey]=0
                totalDiffer[timekey]+=abs(float(price) -float(actualPriceDict[timekey][bestBrandMatch][bestMatch]))
                actualPriceDiff[timekey][bestBrandMatch][bestMatch]=abs(float(price) -float(actualPriceDict[timekey][bestBrandMatch][bestMatch]))
                foundPriceDict[timekey][bestBrandMatch][bestMatch]=float(price)
                timeCounts[timekey]+=1
            #print '------------------------' 

with codecs.open("final.csv", 'wb', 'utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|')
    for key in actualPriceDiff.keys():
        for brand in actualPriceDiff[key].keys():
            for subbrand in actualPriceDiff[key][brand].keys():
                foundPriceDict[key][brand][subbrand]
                actualPriceDiff[key][brand][subbrand]
                foundPriceDict[key][brand][subbrand]
                actualPriceDict[key][brand][subbrand], actualPriceDiff
                writer.writerow([key, brand, subbrand, foundPriceDict[key][brand][subbrand], actualPriceDict[key][brand][subbrand], actualPriceDiff[key][brand][subbrand]])

with codecs.open("data3.csv", 'wb', 'utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|')
    ourBrands=[('BUDWEISER', '6 Pk 12 Oz Glass'),
                ('CORONA LIGHT', '6 Pk 12 Oz Glass'),
                ('CORONA EXTRA', '6 Pk 12 Oz Glass'),
                ('STELLA ARTOIS', '6 Pk 11.2 Oz Glass'),
                ('HOEGAARDEN', '6 Pk 11.2 Oz Glass')
               ]
    beerSet=['_'.join(i) for i in ourBrands]
    beerSet=['_'.join(i.split()) for i in beerSet]
    writer.writerow(['Year']+['_'.join(i.split()) for i in beerSet])
    for timekey in sorted(totalDiffer.keys()):
        
        writer.writerow([timekey, actualPriceDiff[timekey][ourBrands[0][0]].get(ourBrands[0][1],4), actualPriceDiff[timekey][ourBrands[1][0]].get(ourBrands[1][1],4), actualPriceDiff[timekey][ourBrands[2][0]].get(ourBrands[2][1],4), actualPriceDiff[timekey][ourBrands[3][0]].get(ourBrands[3][1],4), actualPriceDiff[timekey][ourBrands[4][0]].get(ourBrands[4][1],4)])


            
