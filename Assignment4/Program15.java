import java.util.Random;
public class SumRandomIntegers {
    public static void main (String [] arg)
    {
        Random randGen = new Random ();
        // identify variables
        int numIntegers = 0;
        int randomInt;
        int sum = 0;
        double average;
        int maxValue = 0;

        // loop to generate random numbers and sum them
        while (numIntegers < 100) {
            randomInt = randGen.nextInt(1000);
            sum = sum + randomInt;
            numIntegers = numIntegers + 1;

            if (randomInt > maxValue) {
                maxValue = randomInt;
            }
        }

        // calculate average
        average = (double)sum/numIntegers;

        // print sum, average and max
        System.out.println("The sum of random 100 integers was " + sum);
        System.out.println("The average of 100 random integers was " + average);
        System.out.println("The largest number among 100 random integers " + maxValue );
    }
}
