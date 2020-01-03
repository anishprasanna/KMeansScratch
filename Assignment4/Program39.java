import java.util.Random;

public class SumRandomIntegers {
    public static void main(String [] args)
    {
        int randomNumSum = 0;
        double randomNumAvg;
        int largestRandomNum = 0;
        int programCounter = 0;
        boolean keepGeneratingNums = true;
        int newestInt;
        //here are the variables I use, They include the sum, largest num, average, a counter, a value for each new number and a boolean for the while loop

        Random randGen = new Random();
        //This is the random number generator used in the program
        while (keepGeneratingNums){
            newestInt = randGen.nextInt(1000);
            //This generates a value from 0 to 999
            randomNumSum = randomNumSum + newestInt;
            programCounter ++;
            //This will add to the sum and the program counter.
            if (newestInt > largestRandomNum){
                largestRandomNum = newestInt;
            }
            //This if statement calculates and stores the largest value of all 100 random numbers
            if (programCounter == 100){
                keepGeneratingNums = false;
            }
            //This block of code tells the while loop when to stop
        }
        randomNumAvg = (double) randomNumSum / programCounter;
        System.out.println("The sum of random 100 integers was " + randomNumSum);
        System.out.printf("The average of 100 random integers was " + "%.2f",randomNumAvg);
        System.out.println("");
        System.out.println("The largest number among 100 random integers was " + largestRandomNum);
        //This block of code prints all of the values that the program just calculated and stored.
    }
}
