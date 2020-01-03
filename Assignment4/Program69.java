import java.util.Random;

public class SumRandomIntegers
{
    public static void main (String[] args)
    {
        Random rand = new Random();
    //variables used
        int temp;
        int sum = 0;
        double counter = 0 ;
        int max = 0;
        double average;


//loops throught the random numbers 100 times
        while (counter < 100)
        {
           //creates a random number and adds it to the sum
            temp =  rand.nextInt(1000);
            sum = sum + temp;
            //checks for maximums
            if(temp>max)
            {
                max = temp;
            }
            //increments counter
            counter++;
        }
        //calculates average and out puts the average, the sum, and the max 
        average = sum/counter;
        System.out.println("The sum of random 100 integers was " + sum);
        System.out.println("The average of 100 random integers was " + average);
        System.out.println("The largest number among 100 random integers was " + max);
    }
}
