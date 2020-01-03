import java.util.Random;

public class SumRandomIntegers {

    public static void main(String [] args){

        Random randGen = new Random();
        int randomSum; // Defines the sum of all the random integers
        double randomAverage = 0.0; // Defines the average of all the random integers
        int randomMax = 0; // Defines the maximum integer produced of all the random integers
        int i = 0; // Loop variable


        randomSum = randGen.nextInt(1000)* 100; // Defines the bound of the random generator as 1000
        System.out.println("The sum of 100 random integers was: " + randomSum); //Outputs sum of random integers


        randomAverage = randomSum/100.0; //Defines the formula for the average of the random integers
        System.out.println("The average of 100 random integers was: " + randomAverage);
        //Outputs the average of the random integers


        randomMax = randGen.nextInt(1000); // Defines the bound of the random generator
        for (i += 0; i < randomMax; i++) {
            if (i ==0) {
                System.out.println("The largest number among 100 integers was: " + randomMax);
            } // Outputs the largest integer generated
            else if(randomSum < randomMax) {
                System.out.println("The largest number among 100 integers was: " + randomMax);
            }

        }






    }
}
