import java.util.Random;
import java.util.Scanner;

public class SumRandomIntegers {
    public static void main (String [] args) {
        Scanner scnr = new Scanner(System.in);
        Random randGen = new Random();
        //Set up scanner and random number generator

        System.out.println("Random Integers =============");
        System.out.println(" ");

        int randNum;
        int largestNum=0;
        int randNumSum;
        double randNumAvg;
        final double TOTAL_NUMS = 100;
        int i;
        //define my variables

        randNumSum = 0;
        for( i = 0 ; i < 100 ; ++i )   {
            randNum = randGen.nextInt(1000);
            randNumSum = randNumSum + randNum;
            if (randNum > largestNum)   {
                largestNum = randNum;
            }
        }
        //Find sum of the 100 random numbers

        randNumAvg = (randNumSum / TOTAL_NUMS);
        //find average of the random numbers

        System.out.println("The sum of random 100 integers was " + randNumSum);
        System.out.println("The average of the random 100 integers was " + randNumAvg);
        System.out.println("The largest number amoung 100 random integers was " + largestNum);

    }
}
