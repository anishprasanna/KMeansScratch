import java.util.Scanner;

public class CountInputLength {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int sum = 0;
        String text;
        System.out.println("Please enter line of text:");
        text = sc.nextLine();
        char[] textarray = text.toCharArray();

        for (int i = 0; i < textarray.length; i++) {

            if (textarray[i] == ' ' || textarray[i] == ',' || textarray[i] == '.') {
                continue;
            }
            else sum++;
            }

            System.out.println("The text length without spaces, periods, or commas is: " + sum);


    }
}
