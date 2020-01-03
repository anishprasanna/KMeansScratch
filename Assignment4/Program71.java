import java.util.Scanner;

public class FizzBuzzGame
{
    public static void main( String [] args)
    {
       //creates and initals variables
        Scanner scnr = new Scanner(System.in);
        int x, y, n;
        System.out.println("Input three integers: ");
        x = scnr.nextInt();
        y = scnr.nextInt();
        n = scnr.nextInt();
    //this loops and prints the numbers and the fizz,buzz, or fizzbuzz
        for(int i =1; i<=n; i++)
        {
            //this checks if it is fizz, buzz, or both and prints the outcome
            if(i%x ==0 && i%y ==0)
                System.out.println("FizzBuzz");
            else if (i%x ==0)
                System.out.println("Fizz");
            else if (i%y == 0 )
                System.out.println("Buzz");
            else
            System.out.println(i);
        }
    }
}
