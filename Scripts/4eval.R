# Here we show an example of how to run the analyses in 6.2 and 6.3, with BOCE and GloVe word vectors.

#Outline:
#0: Set up your environment
#1. load GloVe and BOCE vectors
#2. Run pairwise correlations
#3. load Wiki test set and calculate modularities
#4. load AG news and run text classification example


#TODO: convert to rmarkdown
#--------------------------------------------------------------------------
#Step 0: Set up the following environment

#Download and unzip GloVe vectors - https://nlp.stanford.edu/projects/glove/
glovefilepath = "/path/to/glove.6B.300d.txt"

#Download and unzip BOCE vectors 
bocefilepath = "/path/to/BOCE.English.400K.100d"
bocevocabpath = "/path/to/BOCE.English.400K.vocab.txt"


#This is for Step 3.
#Downlod wikipedia and follow the README instructions to create the train/test pkl files (Note:this is time consuming)
#Within python, load and convert the Wiki test data to csv as follows

{
  #import pickle
  #def save_obj(obj, name ):
  #  with open(name + '.pkl', 'wb') as f:
  #    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
  
  #def load_obj(name ):
  #  with open(name + '.pkl', 'rb') as f:
  #    return pickle.load(f)
  
  #testdata = load_obj("Paragraphdata-testing")
  
  #import  csv
  #with open("testdata.csv","w") as f:
  #  wr = csv.writer(f, delimiter='|')
  #  wr.writerows(testdata)
}

wikitestfilepath = "/path/to/testdata.csv"


#--------------------------------------------------------------------------

## Step 1. load GloVe vectors

#adapted from github tjvananne/process GloVe pre-trained word vector.R
# Define a reading function. Input .txt file, exports list of list of values and character vector of names (words)
proc_pretrained_vec <- function(p_vec, providednames=NULL) {
  
  # initialize space for values and the names of each word in vocab
  vals <- vector(mode = "list", length(p_vec))
  extractednames <- character(length(p_vec))
  
  # loop through to gather values and names of each word
  for(i in 1:length(p_vec)) {
    if(i %% 1000 == 0) {print(i)}
    this_vec <- p_vec[i]
    this_vec_unlisted <- unlist(strsplit(this_vec, " "))
    
    if(is.null(providednames)){ #names are provided inside p_vec
      this_vec_values <- as.numeric(this_vec_unlisted[-1]) 
      this_vec_name <- this_vec_unlisted[1]
      
      vals[[i]] <- this_vec_values
      extractednames[[i]] <- this_vec_name
      
    } else {
      vals[[i]]  <- as.numeric(this_vec_unlisted)  
      }
    }
    
      
  
  # convert lists to data.frame and attach the names
  outputmat <- data.frame(vals)
  if(is.null(providednames)){
    names(outputmat) <- extractednames
  } else {
    names(outputmat) <- providednames
  }
  
  return(outputmat)
}


# here we are reading in the unzipped, raw, GloVe pre-trained word vector object (.txt)
# all you have to change is the file path to where you GloVe object has been unzipped
g6b_300 <- scan(file = glovefilepath, what="", sep="\n")


# call the function to convert the raw GloVe vector to data.frame (extra lines are for wall-time reporting)
t_temp <- Sys.time()
glove.300 <- proc_pretrained_vec(g6b_300)  # this is the actual function call
(t_elap_temp <- paste0(round(as.numeric(Sys.time() - t_temp, units="mins"), digits = 2), " minutes"))

print(dim(glove.300))
rm(g6b_300)



#Load BOCE vectors
BOCEfile <- scan(file= bocefilepath, what="", sep="\n")
BOCEvocab = readLines(bocevocabpath)


BOCE.100 <- proc_pretrained_vec(BOCEfile, BOCEvocab)



#Test encoding data

installAndLoad <- Vectorize(function(...) { #Better installing packages
  loaded <- suppressWarnings(require(..., warn.conflicts = FALSE, quietly=TRUE))
  if(!loaded) {
    install.packages(...,repos = "http://cran.us.r-project.org")
    loaded <- require(..., warn.conflicts = FALSE, quietly=TRUE)
  }
  invisible(loaded)
})


encodetextdata <- function(rawdata, wordvectors){
  installAndLoad("tm")
  installAndLoad("quanteda")
  dtm <- dfm(rawdata, tolower = TRUE)
  convert(dtm, to = "tm")
  dtm = as.matrix(dtm)
  
  
  removerows = which(rowSums(dtm)==0)
  if(length(removerows)>0){
    dtm = dtm[-removerows,]
    rawdata = rawdata[-removerows]
    dtm <- dfm(rawdata, tolower = TRUE)
    convert(dtm, to = "tm")
    dtm = as.matrix(dtm)
  }
  
  #***** Subsetting wordvectors makes it run faster
  v = colnames(dtm)[which(  colnames(dtm) %in% colnames(wordvectors))]
  wordvectors = wordvectors[,v]
  
  
  #build encoded version
  etm = matrix(0, nrow = nrow(dtm), ncol = dim(wordvectors)[1]); colnames(etm) = 1:ncol(etm)
  dim(etm)
  for(i in 1:nrow(dtm)){
    presentwordindex = which(dtm[i,]>0)
    presentnames=c()
    for(j in 1:length(presentwordindex)){
      presentnames = c(presentnames, rep(colnames(dtm)[presentwordindex[j]] , dtm[i,presentwordindex[j]]  ))
    }
    presentnames = presentnames[which(presentnames %in% colnames(wordvectors))] #includes multiples
    
    
    if(length(presentnames)==1)
      etm[i,] = (wordvectors[,presentnames])
    if(length(presentnames)>1)
      etm[i,] = rowSums(wordvectors[,presentnames])/length(presentnames)
  }
  
  #sanity checks
  v = which(rowSums(etm)==0)
  if(length(v)>0 & length(v) != nrow(etm)){
    etm = etm[-v,]
    rawdata = rawdata[-v]
    dtm = dtm[-v,]
  }
  if(dim(etm)[1] != dim(dtm)[1] )
    stop()
  if(dim(dtm)[1] != length(rawdata))
    stop()
  
  correlations = cor(t(etm))
  #diag(correlations) = 0
  
  finaldata = list(rawdata=rawdata, presentnames=presentnames, dtm=dtm, etm=etm, correlations=correlations, thewords=colnames(dtm))
  return(finaldata)
}

myrawdata = c("A feral cat is a cat that lives outdoors and has had little or no human contact" ,
              "They do not allow themselves to be handled or touched by humans, and will run away if they are able" ,
              "They typically remain hidden from humans, although some feral cats become more comfortable with people who regularly feed them" ,
              "Atlanta is the capital of, and the most populous city in, the US state of Georgia" ,
              "With an estimated 2017 population of 486,290, it is also the 39th most populous city in the United States" ,
              "The city serves as the cultural and economic center of the Atlanta metropolitan area, home to 5.8 million people and the ninth largest metropolitan area in the nation." )
encodedboce = encodetextdata(myrawdata, BOCE.100)
encodedglove = encodetextdata(myrawdata, glove.300)

min(encodedglove$correlations)
min(encodedboce$correlations) #BOCE's between-topic correlations are lower


installAndLoad("lattice") 
new.palette=colorRampPalette(c("black","red","yellow","white"),space="rgb")
levelplot(t(encodedglove$correlations),col.regions=new.palette(20), xlab="Sentences", ylab="Sentences")
levelplot(t(encodedboce$correlations),col.regions=new.palette(20), xlab="Sentences", ylab="Sentences") #BOCE's between-topic correlations are lower



#Save for easy reloading
save(BOCE.100, file = "BOCE.100.rData")
save(glove.300, file = "glove.6B.300d.rData")


#-----------------------------------------------------------------------------------------------------------
#Step 2. Pairwise correlations


#boce
colindexes = sample(1:ncol(BOCE.100), 500, replace = FALSE)
u1 = cor(BOCE.100[, colindexes])
z1 = u1[lower.tri(u1)]


#glove
colindexes = sample(1:ncol(glove.300), 500, replace = FALSE)
u2 = cor(glove.300[, colindexes])
z2 = u2[lower.tri(u2)] 



s=matrix(0, ncol = 2, nrow = length(z1))
colnames(s) = c("BOCE", "GloVe")
s[,1] = z1
s[,2] = z2

par(mar=c(3,4,1,1)+0.1)
boxplot(s, ylab="Correlation", outline=FALSE)


apply(s, MARGIN = 2, FUN = "median") #BOCE pairwise correlations are lower (approx 0)




#-----------------------------------------------------------------------------------------------------------
#3. Modularity analyses

modularityhistogram <- function(paragraphs, wordvectors, n=1000){
  #fulltext = c()
  
  results = numeric(n)
  xtr=0
  for(p in 1:length(results)){
    if(p %% 100 ==0)
      print(p)
    
    stopcondition = TRUE
    
    while(stopcondition){
      
      t1 = sample(1:length(paragraphs), size = 1)
      t2 = sample(1:length(paragraphs), size = 1)
      while(t1==t2){
        t2 = sample(1:length(paragraphs), size = 1)
      }
      
      x1 = strsplit(paragraphs[t1], split="\\|")[[1]]
      removes = which(x1== " " | x1 == "")
      if(length(removes)>0)
        x1 = x1[-removes]
      
      x2 = strsplit(paragraphs[t2], split="\\|")[[1]]
      removes = which(x2== " " | x2 == "")
      if(length(removes)>0)
        x2 = x2[-removes]
      
      if(length(x1)>3)
        x1 = x1[1:3]
      if(length(x2)>3)
        x2 = x2[1:3]
      
      rawdata = c(x1, x2)
      
      dv = encodetextdata(rawdata, wordvectors)
      etm = dv$etm
      
      
      stopcondition = any(rowSums(etm)==0) | (length(rawdata)!=6) #should be false to continue
    }
    dim(etm)
    
    #fulltext = c(fulltext, rawdata)
    
    #-----------------------------------------------------------------------------------------------------------
    #trying modularity
    installAndLoad("igraph")
    v= cor(t(etm))
    #v=as.matrix(dist(etm[1:20,], upper = TRUE))
    #v = 0.5 + 0.5*cor(t(etm))
    
    installAndLoad("lattice")  #visualization
    #new.palette=colorRampPalette(c("black","red","yellow","white"),space="rgb")
    #levelplot(t(v),col.regions=new.palette(20), xlab="Sentences", ylab="Sentences")
    
    
    #Classic modularity doesn't accept edge weights. We used the modification in S. Gómez, P. Jensen, and A. Arenas. Analysis of community structure in networks of correlated data.
    #This has little effect, but it's there for completeness
    if(min(v)<0){
      
      w=v
      w[w<0] = 0
      g <-graph_from_adjacency_matrix(w, mode = "undirected", weighted = TRUE)
      m1 = modularity(g, membership =c(rep(1, length(x1)), rep(2, length(x2))) , weights = E(g)$weight )
      
      
      #splits
      x = v
      x[x>0] = 0 ; x = abs(x);
      g <-graph_from_adjacency_matrix(x, mode = "undirected", weighted = TRUE)
      m2 = modularity(g, membership =c(rep(1, length(x1)), rep(2, length(x2))) , weights = E(g)$weight )
      
      
      results[p] = m1 * sum(w)/(sum(x) + sum(w)) - m2 * sum(x)/(sum(x) + sum(w))
      #print(results[p])
      
    } else { #normal modularity
      g <-graph_from_adjacency_matrix(v, mode = "undirected", weighted = TRUE)
      results[p] = modularity(g, membership =c(rep(1, length(x1)), rep(2, length(x2))) , weights = E(g)$weight )
    }
    
    
  }
  return(results)
}


#load data
Wikitestparagraphs = readLines(wikitestfilepath)


#run
BOCEmodularity = modularityhistogram(Wikitestparagraphs, BOCE.100)
glovemodularity = modularityhistogram(Wikitestparagraphs, glove.300)



s=matrix(0, ncol = 2, nrow = length(BOCEmodularity))
colnames(s) = c("BOCE", "GloVe")
s[,1] = BOCEmodularity
s[,2] = glovemodularity

par(mar=c(3,4,1,1)+0.1)
boxplot(s, ylab="Modularity", outline=FALSE)

apply(s, MARGIN = 2, FUN = "median") #BOCE modularity is higher




#-----------------------------------------------------------------------------------------------------------
#3. Classification on AG news
installAndLoad("caret")
installAndLoad("randomForest")

#Download and unzip AG's news corpus: http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html


AGfilepath = "/path/to/newsspace200.xml"
library(XML)
result <- xmlParse(file = AGfilepath)
xml_data <- xmlToList(result) 


#The above is time consuming, so save the intermediate result
save(xml_data, file = "xml_data.rData")


#load intermediate result
setwd(AGfilepath)
load("xml_data.rData")



#Getting the subset of data we want

rawAGdata = c()
ydata = c()
n= 2200# we eventually want 2000 in train and 200 in test
i=0
while(TRUE){
  #xml data arranged as 8*i +offset
  concats = paste(xml_data[[8*i +3]], xml_data[[8*i +6]], sep = " ")
  nexty = xml_data[[8*i +5]]
  
  if(!is.null(concats) & !is.null(nexty)){
    rawAGdata = c(rawAGdata, concats)
    ydata = c(ydata, nexty)
  }
  toremove = which(! ydata %in% c("World", "Sports", "Sci/Tech" ))
  if(length(ydata[-toremove])==n)
    break()
  i=i+1
}
toremove = which(! ydata %in% c("World", "Sports", "Sci/Tech" ))
rawAGdata = rawAGdata[-toremove]
ydata = ydata[-toremove]


rawAGdata = tolower(rawAGdata)


ydata = gsub('[[:punct:] ]+',' ',ydata)
ydata = gsub(' ','',ydata)
ydata = as.factor(ydata)





#Create training/test split
splitindexes = sample(1:length(rawAGdata), 200, replace = FALSE)



encodetextdata <- function(rawdata, wordvectors, splitindexes=NULL, ydata=NULL){
  installAndLoad("tm")
  installAndLoad("quanteda")
  dtm <- dfm(rawdata, tolower = TRUE)
  convert(dtm, to = "tm")
  dtm = as.matrix(dtm)
  
  
  removerows = which(rowSums(dtm)==0)
  if(length(removerows)>0){
    dtm = dtm[-removerows,]
    rawdata = rawdata[-removerows]
    dtm <- dfm(rawdata, tolower = TRUE)
    convert(dtm, to = "tm")
    dtm = as.matrix(dtm)
  }
  
  #***** Subsetting wordvectors makes it run faster
  v = colnames(dtm)[which(  colnames(dtm) %in% colnames(wordvectors))]
  wordvectors = wordvectors[,v]
  
  
  #build encoded version
  etm = matrix(0, nrow = nrow(dtm), ncol = dim(wordvectors)[1]); colnames(etm) = 1:ncol(etm)
  dim(etm)
  for(i in 1:nrow(dtm)){
    presentwordindex = which(dtm[i,]>0)
    presentnames=c()
    for(j in 1:length(presentwordindex)){
      presentnames = c(presentnames, rep(colnames(dtm)[presentwordindex[j]] , dtm[i,presentwordindex[j]]  ))
    }
    presentnames = presentnames[which(presentnames %in% colnames(wordvectors))] #includes multiples
    
    
    if(length(presentnames)==1)
      etm[i,] = (wordvectors[,presentnames])
    if(length(presentnames)>1)
      etm[i,] = rowSums(wordvectors[,presentnames])/length(presentnames)
  }
  
  #sanity checks
  v = which(rowSums(etm)==0)
  if(length(v)>0 & length(v) != nrow(etm)){
    etm = etm[-v,]
    rawdata = rawdata[-v]
    dtm = dtm[-v,]
  }
  if(dim(etm)[1] != dim(dtm)[1] )
    stop()
  if(dim(dtm)[1] != length(rawdata))
    stop()
  
  correlations = cor(t(etm))

  #If the splits are provided, it goes into classification mode, and fits a model
  if(!is.null(splitindexes) & !is.null(ydata)){

    mypreprocsettings = NULL #c("center", "scale") #doesn't seem to matter  NULL
    themetric = "Accuracy"
    control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE,  classProbs=TRUE)
    thentree = 500
    ncores = 10
    
    #split
    testx = etm[splitindexes, ]
    testy = ydata[splitindexes]
    trainx = etm[-splitindexes, ]
    trainy = ydata[-splitindexes]
    
    
    library(doParallel)
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
    
    fit1  <-train(x=trainx, y=trainy, method="rf", metric=themetric, trControl=control, ntree=thentree, 
                  preProc = NULL) 
    stopCluster(cl)
    
    
    finaldata = list(rawdata=rawdata, dtm=dtm, etm=etm, correlations=correlations, thewords=colnames(dtm), testx=testx, testy=testy, trainx=trainx, trainy=trainy, testrawdata = rawdata[splitindexes], trainrawdata = rawdata[-splitindexes], fit1=fit1)
    return(finaldata)
  } else {
    finaldata = list(rawdata=rawdata, dtm=dtm, etm=etm, correlations=correlations, thewords=colnames(dtm))
    return(finaldata)
  }
}


calcrandomnoise <- function(outputmat, fit1, testy, rawdata, splitindexes){
  nresults = c()
  for(nwords in 1:100){
    if(nwords %% 10 ==0)
      print(nwords)
    
    testrawdata = rawdata[splitindexes]
    
    for(i in 1:length(testrawdata)){
      newwords = sample(colnames(outputmat), nwords)
      
      testrawdata[i] = paste(c(testrawdata[i], newwords), collapse = " ")
    }
    
    
    
    dv = encodetextdata(testrawdata, outputmat)
    testxnoise = dv$etm
    dim(testxnoise)
    
    
    
    u=confusionMatrix(predict(fit1, testxnoise), testy)
    nresults[nwords] = u$overall["Accuracy"]
  }
  return(nresults)
}




BOCEdata = encodetextdata(rawAGdata, BOCE.100, splitindexes, ydata)
BOCEnoiseresults = calcrandomnoise(BOCE.100, BOCEdata$fit1, BOCEdata$testy, rawAGdata, splitindexes)

Glovedata = encodetextdata(rawAGdata, glove.300, splitindexes, ydata)
Glovenoiseresults = calcrandomnoise(glove.300, Glovedata$fit1, Glovedata$testy, rawAGdata, splitindexes)



plot(Glovenoiseresults, ylab = "Test set accuracy", xlab = "Number of added noise words", ylim = c(0.35,1), type = "l", col="orange", cex.lab=1.5)
lines(BOCEnoiseresults, col="blue")

legend("topright", legend=c("BOCE", "GloVe"),
       col=c("blue", "orange"), lty=1, lwd = 3, cex=1)




