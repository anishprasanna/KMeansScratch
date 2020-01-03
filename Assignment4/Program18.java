import java.util.Scanner;

public class FizzBuzzGame {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);
        int num1 = 0;
        int num2 = 0;
        int num3 = 0;
        int count= 1;


        System.out.println("Fizz Buzz Game  =============");

        System.out.print("\nInput three integers: ");
        num1 = scnr.nextInt();
        while (num1 < 0 || num1 > 100)//checks if input is valid
        {
            System.out.println("Please enter a number between 1 and 100 inclusively.");
            System.out.print("\nInput three integers: ");
            num1 = scnr.nextInt();
        }
        num2 = scnr.nextInt();
        if (num2 <= num1 || num2 > 100)
        {
            System.out.println("Your second number is either less than /equal to  the first number or greater than 100. Please input a number that is between the first number and 100.");
            num2 = scnr.nextInt();
        }
        num3 = scnr.nextInt();
        if (num3 < num2 || num3 > 100)
        {
            System.out.println("Your third number is either less than the first/second number or greater than 100. Please input a number that is between the second number and 100.");
            num3 = scnr.nextInt();
        }
        while(count <= num3)
        {

            if (count % num1 == 0)
            {
                if (count % num2 == 0)
                {
                    System.out.println("FizzBuzz");
                }
                else
                {
                    System.out.println("Fizz");
                }
            }
            else if (count % num2 == 0)
            {
                if (count % num1 == 0)
                {
                    System.out.println("FizzBuzz");
                }
                else
                {
                    System.out.println("Buzz");
                }
            }
            else
            {
                System.out.println(count);
            }
            count += 1;
        }



    }
}
