package org.learnings.aimodels.sentiments;

import org.apache.camel.CamelContext;
import org.apache.camel.impl.DefaultCamelContext;
import org.learnings.aimodels.sentiments.web.api.SentimentsRoute2;

public class Application {

    public static void main(String[] args) throws Exception {
        CamelContext context = new DefaultCamelContext();
        context.addRoutes(new SentimentsRoute2());

        context.start();
        System.out.println("🚀 Sentiment service running on http://localhost:8080/sentiments");
        Thread.sleep(Long.MAX_VALUE);
        context.stop();
    }
}
