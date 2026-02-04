using Raylib_cs;
using System;
using System.Numerics;
using Chess_Challenge.My_Bot.Ethan_Bot;
using ChessChallenge.API;

namespace ChessChallenge.Application;

public static class EvaluationBar
{
    private const float MinEval = -25f;
    private const float MaxEval = 25f;
    private const float MateThreshold = 1_000_000f; // anything beyond this is considered mate

    // Set this from outside to plug in your evaluation function
    public static Func<Board, float> Evaluate => EthanBot.DisplayEval;

    // Keep track of previous eval for smooth animation
    private static float displayedEval;
    private static float lerpSpeed = 3f; // Higher = faster animation

    public static void Draw(Board board, Rectangle boardRect)
    {
        // Clamp eval from your function
        float targetEval = Evaluate(board);

        // Smoothly interpolate displayed eval towards target, only if not mate
        if (MathF.Abs(targetEval) < MateThreshold)
        {
            targetEval = Math.Clamp(targetEval, MinEval, MaxEval);
            displayedEval += (targetEval - displayedEval) * MathF.Min(lerpSpeed * Raylib.GetFrameTime(), 1f);
        }
        else
        {
            displayedEval = targetEval; // instantly show mate
        }

        float barWidth = boardRect.width * 0.08f;
        float barX = boardRect.x + boardRect.width + boardRect.width * 0.04f;
        float barY = boardRect.y;
        float barHeight = boardRect.height;

        // Background
        Rectangle barRect = new(barX, barY, barWidth, barHeight);
        Raylib.DrawRectangleRec(barRect, Color.DARKGRAY);

        // Fill
        float t = Math.Clamp((displayedEval - MinEval) / (MaxEval - MinEval), 0f, 1f);
        float whiteHeight = barHeight * t;

        Rectangle whiteRect = new(
            barX,
            barY + (barHeight - whiteHeight),
            barWidth,
            whiteHeight
        );
        Raylib.DrawRectangleRec(whiteRect, Color.RAYWHITE);

        // Center line at 0
        float midY = barY + barHeight / 2f;
        Raylib.DrawLine((int)barX, (int)midY, (int)(barX + barWidth), (int)midY, Color.BLACK);

        // Eval number below bar
        string evalText;
        if (MathF.Abs(displayedEval) >= MateThreshold)
        {
            evalText = displayedEval > 0 ? "M+" : "M-";
        }
        else
        {
            evalText = displayedEval.ToString("0.00");
        }

        int fontSize = (int)(barWidth * 0.65f);
        int textWidth = Raylib.MeasureText(evalText, fontSize);
        UIHelper.DrawText(evalText, new Vector2((barX + barWidth / 2 - textWidth / 2), (barY + barHeight) + 18), fontSize, 1, Color.LIGHTGRAY);
    }

    // Optional: reset animation when a new game starts
    public static void Reset()
    {
        displayedEval = 0f;
    }
}
