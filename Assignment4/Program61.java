import java.util.Random;

public class SumRandomIntegers {
    public static void main(String[] args) {
        System.out.println("Random Integers =============");
        //preparing integers for recording a set of random number, the variable for the loop, and the sum by initialized as 0
        Random randomNum = new Random();
        int randomNum1;
        randomNum1 = randomNum.nextInt();
        int max = 0;
        int i;
        int sum = 0;
        //loop to get the 100 number from range 0 to 1000
        for (i = 0; i < 100; ++i) {
            randomNum1 = randomNum.nextInt(1000);
            // ge teh sum by adding each new generated random number to the previous sum
            sum = sum + randomNum1;
            //check and take out teh maximum number among the 100 numbers
            if (randomNum1 > max) {
                max = randomNum1;
            }
        }
        //print out answers and chanign the average into double form.
        System.out.println("The sum of random 100 integers was " + sum);
        double average = (double) sum / 100;
        System.out.printf("The average of 100 random integers was %.2f", average);
        System.out.println("");
        System.out.print("The largest number among 100 random integers was " + max);
    }
}
