import java.util.Scanner;

public class OrderedIntegers {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);
        int num1;
        int num2;
        int num3;

        System.out.print("Enter three integers: ");
        num1 = scnr.nextInt();
        num2 = scnr.nextInt();
        num3 = scnr.nextInt();

        //using nested if statements
        if (num1 < num2 && num1 < num3)
        {
            if (num2 < num3)
            {
                System.out.print(num3 + "  " + num2 + "  " + num1);
            }
            else if (num2 > num3)
            {
                System.out.print(num2 + "  " + num3 + "  " + num1);
            }
            else
            {
                System.out.print(num2 + "  " + num3 + "  " + num1);
            }
        }
        if (num2 < num1 && num2 < num3)
        {
            if (num1 < num3)
            {
                System.out.print(num3 + " " + num1 + " " + num2);
            }
            else if (num1 > num3)
            {
                System.out.print(num1 + " " + num3 + " " + num2);
            }
            else
            {
                System.out.print(num3 + " " + num1 + " " + num2);
            }
        }
        if (num3 < num1 && num3 < num2)
        {
            if (num1 < num2)
            {
                System.out.print(num2 + " " + num1 + " " + num3);
            }
            else if (num1 > num2)
            {
                System.out.print(num1 + " " + num2 + " " + num3);
            }
            else
            {
                System.out.print(num1 + " " + num2 + " " + num3);
            }
        }
        if (num1 == num2 && num2 == num3)
        {
            System.out.print(num2 + "  " + num3 + "  " + num1);
        }
    }
}
