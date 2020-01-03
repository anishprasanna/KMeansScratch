import java.util.Random;

public class SumRandomIntegers {
    public static void main(String[] args) {
        Random rand = new Random();
        int sum;
        double average;
        int largestNum;
        int newInput;

        //Random Integers =============
        System.out.println("Random Integers =============\n");

        //Integers' range is (0 < random number < 1000)



        largestNum = rand.nextInt(1001); // 1st rand number as largest
        sum = largestNum; // add first rand num to sum
        for (int i=0; i< 99; i++){
            newInput = rand.nextInt(1001);

        //Calculate the largest number among 100 random integer
            if(newInput > largestNum){ // if larger than largestNum, set this num to largest
                largestNum = newInput;
            }

        //Calculate the sum of random 100 integers
            sum = sum + newInput;
        }



        System.out.println("The sum of random 100 integers was " + sum);

        //Calculate the  average of 100 random integers
        average = (double)sum / 100;
        System.out.printf("The average of 100 random integers was %.2f\n", average);

        System.out.println("The largest number among 100 random integers was " + largestNum);

    }
}




