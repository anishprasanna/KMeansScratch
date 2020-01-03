import java.util.Random;
public class SumRandomIntegers {
    public static void main (String [] args){

        //declare variables to be used in this program
        Random r = new Random();
        int randomInt = r.nextInt(999);
        int sum = 0;
        double num = 0.0;
        double average;
        int max = 0;

        //compute the sum, average and maximum value of 100 random integers
        System.out.println( "Random Integers===========" );
        while (num <100){
            num += 1;
            sum += randomInt;
            randomInt = r.nextInt(999);
            for (int i = 0; i <= 999; i++){
                if (randomInt >= max){
                    max = randomInt;
                    randomInt = r.nextInt(999);
                }
            }
        }

        //print the result to the screen
        average = sum / num;
        System.out.println( "The sum of random 100 integers was " + sum);
        System.out.println( "The average of 100 random integers was " + average);
        System.out.println( "The larges number among 100 random integers was " + max);

    }
}
