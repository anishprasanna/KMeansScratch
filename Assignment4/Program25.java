import java.util.Random;

public class SumRandomIntegers {
    public static void main(String[] args) {

        // Creates random number generator
        Random randGen = new Random();

        // Declares variables and initializes those that need to be
        int sum = 0;
        double average;
        int max = 0;

        // Writes welcome message
        System.out.println("Random Integers ============");

        // For loop adds 100 random numbers together and uses if-structure to find their max
        for (int i=0; i <100; i++) {

            int randNum = randGen.nextInt(1000);

            sum += randNum;

            if (randNum > max) {
                max = randNum;
            }
        }

        // Computes the average
        average = (double)(sum) / 100;

        // Writes output with proper formatting
        System.out.println("The sum of 100 random integers was " + sum);
        System.out.printf("The average of 100 random integers was %.2f \n", average);
        System.out.println("The largest number among 100 random integers was " + max);
    }
}
