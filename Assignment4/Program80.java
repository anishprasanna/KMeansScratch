import java.util.Scanner;
public class OrderedIntegers {
    public static void main (String [] args)
    {
        Scanner scnr = new Scanner (System.in);
        System.out.println("Order Three Integers =========");

        //ask user for three integers
        System.out.print("Enter three integers: ");

        //assign the values for each integer
        int one = scnr.nextInt();
        int two = scnr.nextInt();
        int three = scnr.nextInt();

        //if else statements
        //print out in descending order

        System.out.print("The integers in decreasing order are: ");

        if ((one >= two) && (one >= three))
        {
            if (two >= three)
            {
                System.out.print(one + " " + two +  " " + three);
            }
            else
            {
                System.out.print(one +  " " + three + " " + two);
            }
        }
        else if ((two >= one) && (two >= three))
        {
            if (one >= three)
            {
                System.out.print(two +  " " + one + " " + three);
            }
            else
            {
                System.out.print(two + " " + three + " " + one);
            }
        }
        else if ((three >= one) && (three >= two))
        {
            if (one >= two)
            {
                System.out.print(three + " " + one + " " + two);
            }
            else
            {
                System.out.print(three + " " + two + " " + one);
            }
        }
    }
}
