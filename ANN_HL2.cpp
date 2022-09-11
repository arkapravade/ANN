#include <bits/stdc++.h>
using namespace std;

#define Number_Pattern 20
#define Number_Input 2
#define Number_Hidden 2
#define Number_Output 1
#define MaxIter 100000

#define rand() ((double)rand()/((double)RAND_MAX+1))
#define test 5

int main(){
    int i,j,k,p,np,op,ranpat[Number_Pattern+1],iteration;
    int Nu_Patt=Number_Pattern,Nu_ip=Number_Input,Nu_Hidd=Number_Hidden,Nu_op=Number_Output;
    double ip[Number_Pattern+1][Number_Input+1];
    double ip_raw[Number_Pattern+1][Number_Input+1];
    double Target[Number_Pattern+1][Number_Output+1];
    double Target_raw[Number_Pattern+1][Number_Output+1];
    double Output_denorm[Number_Pattern+1][Number_Output+1];
    double SumH[Number_Pattern+1][Number_Hidden+1],w_IH[Number_Input+1][Number_Hidden+1],Hidd[Number_Pattern+1][Number_Hidden+1];
    double SumO[Number_Pattern+1][Number_Output+1],w_HO[Number_Hidden+1][Number_Output+1],Output[Number_Pattern+1][Number_Output+1];
    double del_O[Number_Output+1],backprop[Number_Hidden+1],del_H[Number_Hidden+1];
    double del_w_IH[Number_Input+1][Number_Hidden+1],del_w_HO[Number_Hidden+1][Number_Output+1];
    double optimal_HN;
    double error,error_pred,error_pred_total,learning_rate=0.1,alpha=0.1,w_t=0.1;
    // double in1_train[Number_Pattern+1], in2_train[Number_Pattern+1], tgt_train[Number_Pattern+1];
    // double in1_test[Number_Pattern+1], in2_test[Number_Pattern+1], tgt_test[Number_Pattern+1];
    // double maxin1_train=INT_MIN,maxin2_train=INT_MIN,minin1_train=INT_MAX,minin2_train=INT_MAX,maxtgt_train=INT_MIN,mintgt_train=INT_MAX;
    // double maxin1_test=INT_MIN,maxin2_test=INT_MIN,minin1_test=INT_MAX,minin2_test=INT_MAX,maxtgt_test=INT_MIN,mintgt_test=INT_MAX;
    // double maxtraintgt=INT_MIN,mintraintgt=INT_MAX;
    // double maxtesttgt=INT_MIN,mintesttgt=INT_MAX;

    ifstream target;
    ifstream input;
    ifstream input1;
    ifstream input2;
    FILE *w;
    w=fopen("Output.txt","w");
    target.open("target.txt");
    input.open("input.txt");
    input1.open("input1.txt");
    input2.open("input2.txt");

    for(p=1;p<=Nu_Patt;p++){
        for(i=1;i<=Nu_op;i++){
            target>>Target_raw[p][i];   //feeding the raw target data
        }
    }

    // for(int i=1;i<=Nu_Patt;i++){
    //     target>>tgt_train[i];
    //     target>>tgt_test[i];
    //     input1>>in1_train[i];
    //     input2>>in2_train[i];
    // }

    // for(int i=1;i<=Nu_Patt;i++){
    //     maxin1_train = max(maxin1_train, in1_train[i]);
    //     minin1_train = min(minin1_train, in1_train[i]);

    //     maxin2_train = max(maxin2_train, in2_train[i]);
    //     minin2_train = min(minin2_train, in2_train[i]);

    //     maxtgt_train = max(maxtgt_train, tgt_train[i]);
    //     mintgt_train = min(mintgt_train, tgt_train[i]);
    // }

    for(p=1;p<=Nu_Patt;p++){
        for(i=1;i<=Nu_op;i++){
            Target[p][i]=(Target_raw[p][i]-87.21)/(97.9-87.21);     //normalizing the raw target data in a scale of 0-1
            //Target[p][i]=(Target_raw[p][i]-mintgt_train)/(maxtgt_train-mintgt_train);     //normalizing the raw target data in a scale of 0-1
        }
    }

    for(p=1;p<=Nu_Patt;p++){
        for(i=1;i<=Nu_ip;i++){
            input>>ip_raw[p][i];    //feeding the raw input data sets
        }
    }

    for(p=1;p<=Nu_Patt;p++){
        for(i=1;i<=Nu_ip;i++){
            if(i==1){
                // ip[p][i]=(ip_raw[p][i]-minin1_train)/(maxin1_train-minin1_train); //normalizing input 1
                ip[p][i]=(ip_raw[p][i]-20)/(96-20); //normalizing input 1
            }
            else{
                // ip[p][i]=(ip_raw[p][i]-minin2_train)/(maxin2_train-minin2_train); //normalizing input 2
                ip[p][i]=(ip_raw[p][i]-565)/(1400-565); //normalizing input 2
            }
        }
    }

    for(j=1;j<=Nu_Hidd;j++){
        for(i=0;i<=Nu_ip;i++){ 
            del_w_IH[i][j]=0;       //setting the difference value as 0
            w_IH[i][j]=2*(rand()-0.5)*w_t;    //setting the Wih weight with random numbers & initial weight considered
        }
    }
    for(k=1;k<=Nu_op;k++){
        for(j=0;j<=Nu_Hidd;j++){
            del_w_HO[j][k]=0;             //setting the difference value as 0
            w_HO[j][k]=2*(rand()-0.5)*w_t;    //setting the Wih weight with random numbers & initial weight considered
        }
    }
    printf("Iteration\tMean Squared Error(MSE)");
    fprintf(w,"Iteration\tMean Squared Error(MSE)");
     
    for(iteration=0; iteration<MaxIter; iteration++){ 
        error=0;
        for(p=1;p<=Nu_Patt;p++){ 
            ranpat[p]=p;    //storing in an array
        }
        for(p=1;p<=Nu_Patt;p++){
            np=p+rand()*(Nu_Patt+1-p);
            op=ranpat[p];
            ranpat[p]=ranpat[np];
            ranpat[np]=op;      //random patterns are selected in different order in each iteration
        }
        for(np=1;np<=Nu_Patt;np++){ 
            p=ranpat[np];       //choosing the order selected for this iteration
            for(j=1;j<=Nu_Hidd;j++){ 
                SumH[p][j]=w_IH[0][j];  //taking bias between input and hidden layers
                for(i=1;i<=Nu_ip;i++){
                    SumH[p][j]+=ip[p][i]*w_IH[i][j];      //adding after multiplying each input value with weightih
                }
                Hidd[p][j]=1/(1+exp(-SumH[p][j]));  //applying log-sigmoid transfer function
            }
            for(k=1;k<=Nu_op;k++){  
                SumO[p][k]=w_HO[0][k];      //taking bias between input and hidden layers
                for(j=1;j<=Nu_Hidd;j++){
                    SumO[p][k]+=Hidd[p][j]*w_HO[j][k];      //adding after multiplying each input value with weightho
                }
                Output[p][k]=1/(1+exp(-SumO[p][k]));        //applying log-sigmoid transfer function
                error+=0.5*pow((Target[p][k]-Output[p][k]),2)/Number_Pattern;       // MSE
                del_O[k]=(Target[p][k]-Output[p][k])*Output[p][k]*(1-Output[p][k]);
            }
            for(j=1;j<=Nu_Hidd;j++){  //backpropagation
                backprop[j]=0;
                for(k=1;k<=Nu_op;k++){
                    backprop[j]+=w_HO[j][k]*del_O[k];
                }
                del_H[j]=backprop[j]*Hidd[p][j]*(1-Hidd[p][j]);
            }
            for(j=1;j<=Nu_Hidd;j++){     // update weights w_IH
                del_w_IH[0][j]=learning_rate*del_H[j]+alpha*del_w_IH[0][j];
                w_IH[0][j]+=del_w_IH[0][j];
                for(i=1;i<=Nu_ip;i++){ 
                    del_w_IH[i][j]=learning_rate*ip[p][i]*del_H[j]+alpha*del_w_IH[i][j];
                    w_IH[i][j]+=del_w_IH[i][j];
                }
            }
            for(k=1;k<=Nu_op;k ++){    // update weights w_HO
                del_w_HO[0][k]=learning_rate*del_O[k]+alpha*del_w_HO[0][k];
                w_HO[0][k]+=del_w_HO[0][k];
                for(j=1;j<=Nu_Hidd;j++){
                    del_w_HO[j][k]=learning_rate*Hidd[p][j]*del_O[k]+alpha*del_w_HO[j][k];
                    w_HO[j][k]+=del_w_HO[j][k];
                }
            }
        }
        if(iteration%10000==0){
            printf("\n%5d\t\t%f",iteration,error);
            fprintf(w,"\n%5d\t\t%f",iteration,error);
        }
        if(error<0.00006){
            break;
        }
        error=0;
    }
    error_pred_total=0;

    if(iteration>=MaxIter){
        printf("\n\nMaximum number of iterations have been reached");
        fprintf(w,"\n\nMaximum number of iterations have been reached");
        }
    
    printf("\n\nTotal number of iteration: %d\n\nPattern\t",iteration);
    fprintf(w,"\n\nTotal number of iteration: %d\n\nPattern\t",iteration);
    for(i=1;i<=Nu_ip;i++){
        printf("Input-%d\t\t",i);
        fprintf(w,"Input-%d\t\t",i);
    }
    printf("Target value\tOutput value\t");
    fprintf(w,"Target value\tOutput value\t");

    // for(p=1;p<=Nu_Patt;p++){
    //     maxtraintgt = max(maxtraintgt, Output[i][1]);
    //     mintraintgt = min(mintraintgt, Output[i][1]);
    // }

    for(p=1;p<=Nu_Patt;p++){       
    printf("\n%d\t",p);
    fprintf(w,"\n%d\t\t",p);
        for(i=1;i<=Nu_ip;i++){
            printf("%f\t",ip_raw[p][i]);
            fprintf(w,"%f\t\t",ip_raw[p][i]);
        }
        for(k=1;k<=Nu_op;k++){
            Output_denorm[p][k]=Output[p][k]*(97.9-87.21)+87.21;    //de-normalization training outputs
            //Output_denorm[p][k]=Output[p][k]*(maxtraintgt-mintraintgt)+mintraintgt;    //de-normalization training outputs
            printf("%f\t%f\t",Target_raw[p][k],Output_denorm[p][k]);
            fprintf(w,"%f\t\t%f\t\t",Target_raw[p][k],Output_denorm[p][k]);
        }
    }

    printf("\n\n\t\t------TESTING STARTS NOW------\t\t\n");
    fprintf(w,"\n\n\t\t------TESTING STARTS NOW------\t\t\n");

    printf("\nPattern\t");
    fprintf(w,"\nPattern\t");
    for(i=1;i<=Nu_ip;i++){
        printf("Input-%d\t\t",i);
        fprintf(w,"Input-%d\t\t",i);
    }
    printf("Target value\tOutput value\tError in prediction\t");
    fprintf(w,"Target value\tOutput value\tError in prediction\t");

    // for(int i=(Number_Pattern-test); i<=Nu_Patt; i++){
    //     maxtesttgt = max(maxtesttgt, Output[i][1]);
    //     mintesttgt = min(mintesttgt, Output[i][1]);
    // }

    for(int i=(Number_Pattern-test); i<=Number_Pattern; i++){
        printf("\n%d\t",p);
        fprintf(w,"\n%d\t\t",p);
        for(i=1;i<=Nu_ip;i++){
            printf("%f\t",ip_raw[p][i]);
            fprintf(w,"%f\t\t",ip_raw[p][i]);
        }
        for(j=1;j<=Nu_Hidd;j++){
            for(i=1;i<=Nu_ip;i++){
                SumH[p][j]+=ip[p][i]*w_IH[i][j];    //testing on patterns
            }
            Hidd[p][j]=1/(1+exp(-SumH[p][j]));
        }

        for(k=1;k<=Nu_op;k++){
                for(j=1;j<=Nu_Hidd;j++){
                    SumO[p][k]+=Hidd[p][j]*w_HO[j][k];
                }
            Output[p][k]=1/(1+exp(-SumO[p][k]));
        }
        for(k=1;k<=Nu_op;k++){
            Output_denorm[p][k]=Output[p][k]*(92.89-90.1)+90.1;     //de-normalizing testing outputs
            //Output_denorm[p][k]=Output[p][k]*(maxtesttgt-mintesttgt)+mintesttgt;     //de-normalizing testing outputs
            error_pred=fabs(Output_denorm[p][k]-Target_raw[p][k]);
            printf("%f\t%f\t%f\t",Target_raw[p][k],Output_denorm[p][k],error_pred);
            fprintf(w,"%f\t\t%f\t\t%f\t\t",Target_raw[p][k],Output_denorm[p][k],error_pred);
            error_pred_total+=0.5*pow((Output_denorm[p][k]-Target_raw[p][k]),2)/Number_Pattern;  // MSE of error of prediction
        }
    }
    printf("\n\nThe MSE for error of prediction: %f",error_pred_total);
    fprintf(w,"\n\nThe MSE for error of prediction: %f",error_pred_total);
    //double optimal_HN=1+sqrt(Number_Input+Number_Output);
    optimal_HN=17/(2*(Number_Input+Number_Output));
    printf("\n\nThe optimal number of hidden neurons are: %f",optimal_HN);
    fprintf(w,"\n\nThe optimal number of hidden neurons are: %f",optimal_HN);
    return 0;
}