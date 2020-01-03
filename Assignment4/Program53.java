import java.util.Scanner;
public class FizzBuzzGame {
    public static void main(String [] arg)
    {
        Scanner scnr = new Scanner(System.in);              //Set up for user input

        int x;                                      //First integer to be inputted and later to be divided for 'Fizz'
        int y;                                     //Second integer to be inputted and for 'Buzz'
        int n;                                    //Third integer to be inputted for the range of integers to be divided
        int i;                                   //Variable for the for loop; to create range of when to be executed
        double dividedX;                        //Variable to represent the division with integer x
        double dividedY;                       //Variable to represent the division with integer y

        System.out.println("Fizz Buzz Game =============");         //Header
        System.out.println("");                                    //Empty line
        System.out.print("Input three integers: ");               //Statement to be answered by user
        x = scnr.nextInt();                                      //To allow user to input for variable x
        y = scnr.nextInt();                                     //To allow user to input for variable y
        n = scnr.nextInt();                                    //To allow user to input for variable n

        for (i = 1; i <= n; ++i) {                    //Loop to execute for integers within n range
            dividedX = (double) i % x;               //Division for the remainder of integers with variables x and y
            dividedY = (double) i % y;
            if (dividedX == 0 && dividedY != 0) {      //If else statements to determine what to print off of results from modulo
                System.out.println("Fizz");
            } else if (dividedY == 0 && dividedX != 0) {
                System.out.println("Buzz");
            } else if (dividedY == 0 && dividedX == 0) {
                System.out.println("FizzBuzz");
            } else {
                System.out.println(i);
            }
        }


    }
}
