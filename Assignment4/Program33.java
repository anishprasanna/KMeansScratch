import java.util.Scanner;
import java.util.Random;
public class SumRandomIntegers {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);
        Random rnum = new Random();

        System.out.println("Random Integers ==============\n");

        int sum=0;
        int largest=0;

        for(int i =0; i < 101; i++){
            int n = rnum.nextInt(1000);
            sum += n;
            if(n > largest ){
                largest = n;
            }
        }

        System.out.println("The sum of random 100 integers was " +sum);
        System.out.println ("The average of 100 random integers was " + sum/100.0);
        System.out.println ("The largest number among 100 random integers was " +largest);

    }
}
