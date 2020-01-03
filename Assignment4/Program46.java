import java.util.Scanner;
public class CountInputLength {
    public static void main (String []args ){
        Scanner sc = new Scanner(System.in);

        //declare variables to be used in this program
        String userInput;
        char c;
        int length = 0;
        int i;

        //prompt the user to enter a line of text
        System.out.println( "Count Input Length=================" );
        System.out.print( "Enter a line of text: " );
        userInput = sc.nextLine();

        //count the text length
        for (i = 0; i < userInput.length(); i+=1){
            c = userInput.charAt(i);
            if ( (c >= 65) && (c <= 122) ){
                length += 1;
            }
        }

        //print the length to the screen
        System.out.println( "The text length without spaces, periods, or commas is: " + length);
    }
}
