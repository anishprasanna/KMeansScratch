import java.util.Random;

import java.util.Random;
public class SumRandomIntegers {
    public static void main(String [] arg)
    {
        Random randomInt = new Random();
        int randNum = 0;
        double randAverage = 0;
        int i;
        int randSum = 0;
        int randMax = 0;
        final int ONE_HUNDRED = 100;

        System.out.println("Random Integers =========");
        System.out.println();

        for (i = 0; i < 100; i++) {
            randNum = randomInt.nextInt(1000);
            randSum = randSum + randNum;
            if (randNum > randMax) {
                randMax = randNum;
            }
        }

        randAverage = (randSum / ONE_HUNDRED);

        System.out.println("The sum of random 100 integers was " + randSum);
        System.out.println("The average of 100 random integers was " + randAverage);
        System.out.println("The largest number among 100 random integers was " + randMax);
    }
}
