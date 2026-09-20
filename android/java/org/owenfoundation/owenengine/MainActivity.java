package org.owenfoundation.owenengine;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;
import android.view.Gravity;

/** Minimal launcher screen: the engine itself is used from chess GUIs. */
public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView view = new TextView(this);
        view.setText("Owen 2 engine (v2.1.0, net baked in).\n\n"
                + "This app provides the engine to chess GUIs such as DroidFish:\n"
                + "open the GUI, go to Manage Chess Engines, and pick Owen 2.\n\n"
                + "https://github.com/Owen-Foundation/Owen");
        view.setGravity(Gravity.CENTER);
        view.setPadding(48, 48, 48, 48);
        view.setTextSize(16);
        setContentView(view);
    }
}
