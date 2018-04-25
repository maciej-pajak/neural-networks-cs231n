package pl.maciejpajak.cifar.util;

import javafx.util.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * A class used to quickly display images.
 * Inspired by org.knowm.xchart.SwingWrapper
 */
public class ImageDisplayer {

    private final List<Pair<String, BufferedImage>> images;
    private int rows;
    private int columns;
    private String title;

    /**
     * Create new ImageDispalyer.
     * @param title - window title.
     * @param rows - number of rows in diplay grid.
     * @param columns - number of columns in display grid.
     */
    public ImageDisplayer(String title, int rows, int columns) {
        this(new ArrayList<>(), title, rows, columns);
    }

    /**
     * Create new ImageDispalyer.
     * @param images - list of images with labels.
     * @param title - window title.
     * @param rows - number of rows in diplay grid.
     * @param columns - number of columns in display grid.
     */
    public ImageDisplayer(List<Pair<String, BufferedImage>> images, String title, int rows, int columns) {
        this.images = images;
        this.rows = rows;
        this.columns = columns;
        this.title = title;
    }

    /**
     * Add image to this image displayer.
     * @param title - image title.
     * @param image - image.
     */
    public void addImage(String title, BufferedImage image) {
        images.add(new Pair<>(title, image));
    }

    /**
     * Shows window with images and labels.
     * @return - JFrame with images.
     */
    public JFrame show() {
        final JFrame frame = new JFrame(title);

        javax.swing.SwingUtilities.invokeLater(() -> {

            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            frame.getContentPane().setLayout(new GridLayout(rows, columns));

            images.forEach((pair) -> {
                frame.add(new JLabel(pair.getKey(), new ImageIcon(pair.getValue()), SwingConstants.LEFT));
            });

            // Display the window.
            frame.pack();
            frame.setVisible(true);

        });
        return frame;
    }
}
