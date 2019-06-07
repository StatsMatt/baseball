## Web Scraping for Baseball Predictive Model ##
## Author: Matt Misterka ##
## Date: 6/7/2019 ##

# Program Description: This program grabs data from baseball-reference: https://www.baseball-reference.com/
# Specifically, it scapes game-by-game data for pitchers Nolan Ryan and Tommy John.
# The end-goal is to use the dataset to build a predictive model.
# A model built on a training set that classifies the pitcher for each game in a test set.

# Load in the libraries
library(rvest)
library(stringr)
library(tidyverse)
library(plyr)
library(GGally)
library(caret)
# set the directory
setwd("C:\\Users\\Matt\\Documents\\Statistics\\Stat_Learning")

# these are the urls for the first season of each player
base_url <- "https://www.baseball-reference.com/players/gl.fcgi?id=ryanno01&t=p&year=1966"
base_url2 <- "https://www.baseball-reference.com/players/gl.fcgi?id=johnto01&t=p&year=1963"

# create a dataset of urls to pull data from Nolan Ryan
year<-seq(1966,1993,1)
pages<-data.frame(year)
pages$year<-as.character(pages$year)
pages$link<-NA
pages$link[1]<-base_url

for (i in 2:nrow(pages)){
  
  pages$link[i]<-gsub("1966$", pages$year[i], base_url) 
  
}

# grab Nolan Ryan data
game_data<-NULL
for (j in 1:nrow(pages)){
  
  webpage <- read_html(pages$link[j])
  ryan_table <- html_table(webpage, fill = T)
  ryan_pergame<-data.frame(ryan_table)
  game_data<-rbind.fill(game_data,ryan_pergame)
  
}

# format the result of the game 
for (k in 1:nrow(game_data)){
  
  game_data$Result[k]<-substring(game_data$Rslt[k], 1, 1)
}

# remove "summary" rows
game_data$Gcar<-as.numeric(game_data$Gcar)
game_data<-game_data[!is.na(game_data$Gcar), ]

# take out incomplete variables
game_data<-game_data[,-c(1:6,8:10,27:29,33,40:48)]
game_data[,c(2:26)] <- sapply(game_data[,c(2:26)],as.numeric)

# Repeat above for Tommy John
year2<-seq(1963,1989,1)
pages2<-data.frame(year2)
pages2$year2<-as.character(pages2$year2)
pages2$link<-NA
pages2$link[1]<-base_url2

for (i in 2:nrow(pages2)){
  
  pages2$link[i]<-gsub("1963$", pages2$year2[i], base_url2) 
  
}

game_data2<-NULL
for (m in 1:nrow(pages2)){
  
  webpage <- read_html(pages2$link[m])
  john_table <- html_table(webpage, fill = T)
  john_pergame<-data.frame(john_table)
  game_data2<-rbind.fill(game_data2,john_pergame)
  
}

game_data2$Gcar<-as.numeric(game_data2$Gcar)
game_data2<-game_data2[!is.na(game_data2$Gcar), ]

for (n in 1:nrow(game_data2)){
  
  game_data2$Result[n]<-substring(game_data2$Rslt[n], 1, 1)
}

game_data2<-game_data2[,-c(1:6,8:10,27:29,33,40:48)]
game_data2[,c(2:26)] <- sapply(game_data2[,c(2:26)],as.numeric)

# create Name variable
game_data$Name<-"Nolan"
game_data2$Name<-"Tommy"

# create baseball dataset of ~1500 games
baseball<-rbind(game_data,game_data2)

# remove observation with one tie
baseball<-baseball %>% 
  filter(Result != "T")

# format variables correctly
baseball$Opp<-as.factor(baseball$Opp)
baseball$Result<-as.factor(baseball$Result)
baseball$Name<-as.factor(baseball$Name)

# randomize the rows
set.seed(741776)
baseball <- baseball[sample(1:nrow(baseball)), ]

# sample 20 observations for the test set
x<-1:1567

set.seed(741776)
test<-sample(x,20,replace = F)

# training and test sets
baseball.test<-baseball[test, ]
baseball.train<-baseball[-test, ]

save(baseball.test,file="baseball_test.Rda")
save(baseball.train,file="baseball_train.Rda")

# Dummy variables for opponents chosen by Lasso
baseball.train$MON<-ifelse(baseball.train$Opp=="MON",1,0)
baseball.train$CAL<-ifelse(baseball.train$Opp=="CAL",1,0)
baseball.train$LAD<-ifelse(baseball.train$Opp=="LAD",1,0)
baseball.train$OAK<-ifelse(baseball.train$Opp=="OAK",1,0)
baseball.train$WSA<-ifelse(baseball.train$Opp=="WSA",1,0)

cols <- c("MON","CAL","LAD","OAK","WSA")
baseball.train[cols] <- lapply(baseball.train[cols],as.factor)
