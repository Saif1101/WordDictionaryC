#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

void viewDict(void);
void addition(void);
void search(void);
void quiz(void);

char words[100][100],meanings[100][100];
int lastEmptyIndex=0,lastEmptyIndexMeaning=0;
int userChoice = 0;


int main(void) {

  while(userChoice!=5){
  
  printf("___Welcome To Your Dictionary____\n" 
  "   Press 1 to View Your Dictionary   \n"
  "   Press 2 to Add Words   \n"
  "   Press 3 to Search for a Word    \n"
  "   Press 4 for a Quiz              \n"    
  "   Press 5 to exit -->  ");
  scanf("%d",&userChoice);
  while ((getchar()) != '\n'); 
  
  if(userChoice == 1 && lastEmptyIndex == 0){
    printf("\n ___The Dictionary is empty___ \n \n");
  }
  else if(userChoice == 5){
    printf("Thank You!"); 
    break;
  }
  else if(userChoice == 1){
    viewDict();
  }
  else if(userChoice ==2){
    addition();
  }

  else if(userChoice == 3){
    search();
  }
  else if(userChoice == 4){
    quiz();
  }
  else{
    printf("Invalid option. Please enter correct option.");
  }

  }
}

void viewDict(void) {
  printf("\nThe Dictionary : \n \n");
    for(int i = 0; i < lastEmptyIndex; i++)
    {
        printf("%s:-\n ", words[i]);
        printf("\t%s\n", meanings[i]);

    }
}


void addition(void) {
  
  char wordToBeAdded[12],auxillarySortingArray[50];
  printf("Please Enter A Word  ");
  fgets(wordToBeAdded,50,stdin);
  strcpy(words[lastEmptyIndex],wordToBeAdded);
  lastEmptyIndex++;

  //Positioning the added word (lexicographic order)

  for (int i = 0; i < lastEmptyIndex; ++i) {
        for (int j = i + 1; j < lastEmptyIndex; ++j) {
            if (strcmp(words[i], words[j]) > 0) {
                strcpy(auxillarySortingArray, words[i]);
                strcpy(words[i], words[j]);
                strcpy(words[j], auxillarySortingArray);
            }
        }
    }
  
  // Finding the final index of the word so that its meaning can be assigned to the same place in the meaning array.

  int indexForMeaning = 0;
  
  if(lastEmptyIndex == 1){
    indexForMeaning =0;
  }
  else{
    for (int i= 0; i<lastEmptyIndex;i++){
        if(strcmp(wordToBeAdded,words[i])==0){
          indexForMeaning = i;
        }
    }
  }

  char meaningOfWord[50];
  printf("Enter The Word's Meaning:- ");
  fgets(meaningOfWord,50,stdin);
  //inserting the meaning at indexForMeaning
  for(int i=lastEmptyIndex; i>=indexForMeaning; i--)
        {
          strcpy(meanings[i],meanings[i-1]);
        }
        
        /* Insert new element at given position and increment size */
        strcpy(meanings[indexForMeaning],meaningOfWord); 

  
}

void search(void){
    char wordToBeSearched[15];int flag=0;
    printf("\n Enter the word you want to search for_  ");
    fgets(wordToBeSearched,sizeof(wordToBeSearched),stdin);
    for(int i = 0; i<lastEmptyIndex;i++){
      if(strcmp(wordToBeSearched, words[i])==0){
        flag=1;
        printf("%s:-\n ", words[i]);
        printf("\t%s\n", meanings[i]);
        break;
      }
    }
    if(flag==0){
      printf("\n Word not found! \n \n");
    }
}

void quiz(void){
  int score =0; 
  int rand_num;
  char GuessMeaning[50];
  srand(time(0));
  rand_num = (rand() % (lastEmptyIndex - 0 + 1)) + 0;
  if(lastEmptyIndex==0){
    printf("Please enter a few words first!");
  }
  else{
    printf("____The word is -----> %s", words[rand_num]);
    printf("Enter its meaning -->");
    fgets(GuessMeaning,50,stdin);
    if (strcmp(GuessMeaning, meanings[rand_num])==0){
      printf("You got the meaning right!");
    }
    else{
      printf("Oops! Better luck next time!");
    }
    }
  }


