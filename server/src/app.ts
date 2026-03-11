import express from "express";
import type { Application, Request, Response } from "express";
import cors from "cors";

import authRoutes from "./routes/auth.routes.ts";
import chatRoutes from "./routes/chat.routes.ts";
import errorHandler from "./middleware/error-handler.middleware.ts";

const app: Application = express();

/* CORS Configuration */
app.use(
  cors({
    origin: [
      "http://localhost:5173",
      "https://edureach-pb2z.vercel.app",
      "https://edureach-five.vercel.app",
    ],
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "PATCH"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

/* Middleware */
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

/* Routes */
app.use("/api/auth", authRoutes);
app.use("/api/chat", chatRoutes);

/* 404 Handler */
app.use((_req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    message: "Route not found.",
  });
});

/* Global Error Handler */
app.use(errorHandler);

export default app;