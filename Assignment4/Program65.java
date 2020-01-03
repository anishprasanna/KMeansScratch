import java.util.Random;

public class SumRandomIntegers {
    public static void main(String[] args) {
        // Output program title
        System.out.println("Random Integers =============\n");

        // Declare variables and create random number generator
        int num;
        double sum = 0;
        int max = 0;
        final int ONE_HUNDRED_NUMBERS = 100;
        Random randGen = new Random();

        // Generate 100 random numbers and calculate sum and max
        for (int i = 0; i < 100; i++) {
            num = randGen.nextInt(1000);
            sum += num;                             // Calculate sum
            if (num > max) {                        // Calculate max
                max = num;
            }
        }

        // Output calculations for sum, average, and maximum integer
        System.out.println("The sum of random 100 integers was " + (int) sum);
        System.out.println("The average of 100 random integers was " + sum / ONE_HUNDRED_NUMBERS);
        System.out.println("The largest number among 100 random integers was " + max);
    }
}
