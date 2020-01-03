import java.util.Scanner;
import java.util.Random;
public class SumRandomIntegers {
    public static void main(String[] args) {
        Random rnum = new java.util.Random();
        Scanner scnr = new java.util.Scanner(System.in);
        //I defined num1, num2, and num3 as the result of a random number generator, bounded by 0 and 1000.
        int num1 = rnum.nextInt(1000);
        int num2 = rnum.nextInt(1000);
        int num3 = rnum.nextInt(1000);
        double avg = (num1 + num2 + num3) / 3;
        double sum = (num1 + num2 + num3);
        int i = 0;
        int max = 0;
        //This while loop will output the maximum of the numbers.
        while (i < 2) {
            if ((num1 >= num2) && (num1 >= num3)) {
                max = num1;
            } else if ((num2 >= num1) && (num2 >= num3)) {
                max = num2;
            } else {
                max = num3;
            }
            i++;
        }
            System.out.println("The sum of the three numbers was " + sum);
            System.out.println("The average of the three numbers was " + avg);
            System.out.println("The largest of the three numbers was " + max);

        }
    }






