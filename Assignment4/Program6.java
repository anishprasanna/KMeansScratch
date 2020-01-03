import java.util.Scanner;
import java.util.Random;
public class SumRandomIntegers {
    public static void main (String [] args){
        Scanner scnr = new Scanner(System.in);
        Random rndm = new Random();

        //Declare the variables
        int integer;
        int integer2 = 0;
        double counter;
        int sum = 0;
        double average = 0.00;
        int maximum = 0;

        //Create a loop that picks 100 different integers and progressively updates the three values
        for (counter = 1; counter <= 100; counter++){
            integer = rndm.nextInt(1000);
            sum = sum + integer;
            average = sum / counter;
            if (integer > integer2){
                maximum = integer;
                integer2 = integer;
            }

        }

        //Print the the sum, the average, and the maximum of the randomly picked values
        System.out.println("Random Integers =============");
        System.out.println();
        System.out.println("The sum of random 100 integers was " + sum);
        System.out.print("The average of 100 random integers was ");
        System.out.printf("%.2f", average);
        System.out.println();
        System.out.println("The largest number among 100 random integers was " + maximum);
    }


}
