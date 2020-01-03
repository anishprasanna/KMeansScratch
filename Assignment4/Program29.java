import java.util.Random;
public class SumRandomIntegers {
    public static void main (String[] arg){
        //create variables
        int sumNumbers = 0;
        int largestNum = 0;
        int temporary = 0;
        //create for loop with a limit of 100
        for(int i = 0; i < 100; i++){
            Random randGen = new Random();
            temporary = randGen.nextInt(1000);
            //test to see if random number is greater than the previous largest number
            if(temporary > largestNum){
                largestNum = temporary;
            }
            //add the random num to total sum of numbers
            sumNumbers += temporary;
        }
        //calculate the average of the numbers
        double avgNumbers = sumNumbers/100.0;
        //Print the sum, average, and largest int of the random numbers
        System.out.println("The sum of 100 random integers is: " + sumNumbers);
        System.out.println("The average of 100 random integers is: " + avgNumbers);
        System.out.println("The largest number among 100 random integers is: " + largestNum);
    }
}
