import java.util.Random;
public class SumRandomIntegers {
    public static void main(String [] args)
    {
        Random randGen = new Random();
        //Define variables and compute sum
        int numIntegers = 0;
        int sum = 0;
        int newInt;
        int max = 0;
        double average;

        //Generate 100 random integers and find their sum and max
        while ( numIntegers < 100 ) {
            newInt = randGen.nextInt(1000);
            sum = sum + newInt;
            numIntegers += 1;
            if (newInt > max) {
                max = newInt;
            }
        }

        //Compute average and output sum, average, and max
        average = (double)sum / numIntegers;
        System.out.println("The sum of 100 random integers was " + sum);
        System.out.printf("The average of 100 random integers was %.2f\n", average);
        System.out.println("The largest number of 100 integers was " + max);

    }
}
