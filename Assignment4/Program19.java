import java.util.Random;

public class SumRandomIntegers {
    public static void main(String [] arg){
        Random numGenerator = new Random();
        double sum = 0;
        int num = 0;
        int max = 0;
        double average;
        int count = 0;
        System.out.println("Random Integers =============");
        System.out.println(" ");
        while(count <101)
        {
            num = numGenerator.nextInt(1000);
            sum = sum + num;

            if(num > max)//compares new genreated number to max.
            {
                max = num;//assigns num to new max if its greater than the current max
            }

            count += 1;
        }

        System.out.println("The sum of random 100 integers was " + sum);
        average = sum/100;
        System.out.print("The average of random 100 integers was ");
        System.out.printf("%.2f", average);
        System.out.println("\nThe largest number among the 100 random integers was " + max);
    }

}
