# sys6018-competition-blogger-characteristics

#Members: 

Ning Han ,	Varshini Sriram ,	Henry Tessier ,	Jingnan Yang

#Goal: 
 
 Build a parametric model : linear regression model and apply cross validation. 
 
#Process: 
  1) Data Cleaning and Feature Engineering: 
     There is no missing values in this data but we need to transform text to other numerical features. 
     The features created based on text: 
     A. Number of urls: the number of urls contained in each text post. 
     B. Number of hashtag: the number of hashtags contained in each text post. 
     C. Number of misspelled word: the percentage of misspelled word in each post. (the number of missepelled word/ the number of word)
     D. Number of Punctuation : the number of punctuation in each post. 
     E. Number of Uppercasewords: the number of uppercasewords in each post. 
     F. Number of characters: the number of characters in each post: 
     G: TF-IDF
    
     Other features generated based on other variables: 
     H. Number of posts: number of posts each unique user posted.
     I. Day of week: the weekday for each post date. 
     Based on the understanding of the problem, we think sign is not important to predict age, so we didn't include the feature. 
     The other features in the model are: 
     J: Topics
     K: Gender
     Because both of the are nominal variables, we create dummy varaible for both of them. (also the H: Day of Week). 
   2) Variable Transformation 
      We normalize age, number of urls, number of hashtag, number of Puncuation and several other variables based on their skewness. 
   3) Feature selection: 
      We firstly split our train data to train and test subsets. 
      RFECV: We firstly tried RFECV method, the result is to keep all the varialbes, so we keep all the variables. We calculated the mse for this model. 
      LassoCV: We then started to use LassoCV method to build the second model and calculated the mse based on this model. 
      Lasso: At the end we used lasso to build the thrid model and calculated the mse based on this model. 
      We picked the best model based on their mse score.
#Contributions of each member : 
Ning Han: A,B,C,D,E,H,I,J,K feature and the linear model 
Varshini Sriram: G and model improvements 
Henry Tessier: Number of characters, the setence token and word token package set up and final report          
      
#Results: 
   The best score: 4.89836

#Things can be explored later: 
    1) Genrate more features based on the available data set and further examine current feature effectiveness. 
    2) Based on the distribution of age, the linear regression model doesn't seem like to be the best fit. Maybe try Knn in the future. 
    3) Cross Validation: we used the cross validation for our feature selection, we can also use K-fold cross validation to evalute other model . 
    


      
      
   
  
