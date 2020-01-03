import java.util.Scanner;

public class FizzBuzzGame {
    public static void main(String[] args) {
        System.out.println("Fizz Buzz Game =============");
        // set up variables to record the input integer given by the user
        System.out.println("Input three integers:");
        Scanner scnr = new Scanner(System.in);
        int FirstNum = scnr.nextInt();
        int SecondNum = scnr.nextInt();
        int ThirdNum = scnr.nextInt();
        int i;
        if ( 1 <= FirstNum && FirstNum <SecondNum && SecondNum<= ThirdNum) {
            // loop to calculate if the number tange from 0 to the third number  can modular the number given by the user with the result of 0.
            for (i = 1; i <= ThirdNum; i++) {
                //If could both modular second and first number with result as 0, it is FizzBuzz
                if (i % (SecondNum * FirstNum) == 0)
                    System.out.println("FizzBuzz");
                    //If could do with only first number, answer is Fizz
                else if ((i % FirstNum) == 0)
                    System.out.println("Fizz");
                    //If could modular the second number only, type Buzz
                else if ((i % SecondNum) == 0)
                    System.out.println("Buzz");
                    // If can't do with any number, then type out the original number
                else
                    System.out.println(i);
            }
        }
// if the user type in numbers that do not follow the sequence of 1 ≤ first number  < second number  ≤ third number ≤ 100, show the reader what should they do.
        else {
                System.out.println("The value of three number should be: 1 ≤ first number < second number ≤ third number ≤ 100");
            }
        }
    }
