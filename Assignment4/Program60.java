import java.util.Scanner;
import java.util.Random;
public class SumRandomIntegers {
    public static void main (String[]args){
        System.out.println("Random Integers\n");

        //Number generator set up and variables
        Random generator = new Random();
        int totalNum = 0;
        int sum = 0;
        int largestNum = 0;

        //Use loop to generate 100 random integer, calculating the sum and storing the largest value found
        while (totalNum < 100){
            int number = generator.nextInt(1000);
            if (number > largestNum){
                largestNum = number;
            }
            totalNum++;
            sum = sum + number;
        }

        //Output the sum, the average and the largest number
        double average = sum/(double)totalNum;
        System.out.println("The sum of random 100 integers was " + sum);
        System.out.println("The average of 100 random integers was " + average);
        System.out.println("The largest number among 100 random integers was " + largestNum);
    }
}
