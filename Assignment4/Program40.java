import java.util.Scanner;
public class CountInputLength {
    public static void main(String [] arg)
    {
        Scanner sc = new Scanner(System.in);

        String userString;
        int stringLength = 0;
        int i = 0;

        System.out.println("Count Input Length ========");
        System.out.println();
        System.out.print("Enter a line of text: ");
        userString = sc.nextLine();
        stringLength = userString.length();
        System.out.print("The text length without spaces, periods, or commas is: ");

        for (i = 0; i < userString.length(); i++)
        { if (userString.charAt(i) == ' ')
        {
            stringLength--;
        }
        else if (userString.charAt(i) == ',')
        {
            stringLength--;
        }
        else if (userString.charAt(i) == '.')
        {
            stringLength--;
        }
        }

        System.out.print(stringLength);
    }
}
