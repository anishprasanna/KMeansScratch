import java.util.Scanner;
public class OrderedIntegers {
    public static void main(String [] args)
    {
        Scanner scnr = new Scanner(System.in);

        //Declare variables
        int firstInteger;
        int secondInteger;
        int thirdInteger;

        /*
        Print the purpose of the program then prompt the input (the request for the three integers
        will appear differently from the assignment but we were told in class that prompting the input would not have been graded
        */
        System.out.println("Order Three Integers ==========");
        System.out.println();
        System.out.println("Enter the first integer: ");
        firstInteger = scnr.nextInt();
        System.out.println("Enter the second integer");
        secondInteger = scnr.nextInt();
        System.out.println("Enter the third integer: ");
        thirdInteger = scnr.nextInt();

        /*
        Print the three integers in decreasing order depending on the
        case (seven different cases counting also the case in which the three integers are all the same number)

         */
        if((firstInteger == secondInteger)&&(secondInteger == thirdInteger )){
            System.out.println("The integers are all equal to each other");
        }
        else if( (firstInteger >= secondInteger)&&(secondInteger >= thirdInteger)){
            System.out.println("The integers in decreasing order are: " + firstInteger + " " + secondInteger + " " + thirdInteger);
        }
        else if((secondInteger >= firstInteger)&&(firstInteger >= thirdInteger) ){
            System.out.println("The integers in decreasing order are: " + secondInteger + " " + firstInteger + " " + thirdInteger);
        }
        else if((firstInteger >= thirdInteger)&&(thirdInteger >= secondInteger)){
            System.out.println("The integers in decreasing order are: " + firstInteger + " " + thirdInteger + " " + secondInteger);
        }
        else if((secondInteger >= thirdInteger)&&(thirdInteger >= firstInteger)){
            System.out.println("The integers in decreasing order are: " + secondInteger + " " + thirdInteger + " " + firstInteger);
        }
        else if((thirdInteger >= firstInteger)&&(firstInteger >= secondInteger)){
            System.out.println("The integers in decreasing order are: " + thirdInteger + " " + firstInteger + " " + secondInteger);
        }
        else if((thirdInteger >= secondInteger)&&(secondInteger >= thirdInteger)){
            System.out.println("The integers in decreasing order are: " + thirdInteger + " " + secondInteger + " " + thirdInteger);
        }



    }
}
