import java.util.Random;

public class SumRandomIntegers {
    public static void main (String [] args)
    {
        // create random number generator
        Random randomGen = new Random();

        // generate 100 random integers within the range x >= 0 and x < 1000

        int randomNum;             // initialized with loop
        int i;                     // initialized with loop
        int randomSum = 0;
        int maxNum = 0;
        final int NUM_VALUE = 100; // there will be 100 numbers by the end of the loop

        for (i = 0; i <= 100; i++) {
            randomNum = randomGen.nextInt(1000); // generates a random number between 0 and 1000
            randomSum += randomNum;                      // adds each new random number to the sum

            if (randomNum > maxNum) {                    // assigns the highest value integer as the maximum value
                maxNum = randomNum;
            }
        }

        // calculate the sum, average, and maximum of the one hundred random integers and output the values
        System.out.println("The sum of the 100 random integers was " + randomSum);
        System.out.printf("The average of 100 random integers was %.2f" , (double) randomSum / NUM_VALUE); // cast as double to avoid integer division and round to 2 decimal places
        System.out.println();
        System.out.println("The largest among 100 random integers was " + maxNum);

    }
}
