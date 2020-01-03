import java.util.Scanner;
public class FizzBuzzGame {
    public static void main(String[] arg) {
        int firstNum;
        int secondNum;
        int thirdNum;
        int num = 1;


        Scanner scnr = new Scanner(System.in);
        System.out.print("Enter three integers: ");
        firstNum = scnr.nextInt();
        secondNum = scnr.nextInt();
        thirdNum = scnr.nextInt();


        while (num <= thirdNum) {
            if ((num % firstNum) == 0) {
                if (num % secondNum == 0) {
                    System.out.println("FizzBuzz");
                }
                else {
                    System.out.println("Fizz");
                }
            }
            else  if ((num % secondNum) == 0) {
                System.out.println("Buzz");
            }
             else {
                System.out.println(num);
            }
            num = num + 1;
        }

    }

}

