import java.util.Scanner;
public class FizzBuzzGame {
    public static void main(String[] arg) {
        Scanner scnr = new Scanner(System.in);

        int i;
        int userNum1 = 1;
        int userNum2 = 1;
        int userNum3 = 1;

        System.out.println("Fizz Buzz Game =========");
        System.out.println();
        System.out.print("Input three integers: ");
        userNum1 = scnr.nextInt();
        userNum2 = scnr.nextInt();
        userNum3 = scnr.nextInt();

        for (i = 1; i <= userNum3; ++i) {
            if (userNum3 >= 100 || userNum1 <= 1) {
                System.out.println("Enter integer within bounds");
                break;
            }
                if ((i % userNum1 == 0) && (i % userNum2 == 0))
                {
                    System.out.println("FizzBuzz");
                }
            else if ((i % userNum1) == 0) {
                System.out.println("Fizz");
            }
            else if ((i % userNum2) == 0) {
                System.out.println("Buzz");
            }
            else
                System.out.println(i);
            }
        }
    }


