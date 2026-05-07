# vLLM TPU Inference: AI Engineering Standards

> **CRITICAL INSTRUCTION TO ALL AI AGENTS:** 
> You are operating in a production-grade enterprise repository. Before generating any code or architectural changes, you MUST adhere to the following strict engineering standards. Do not optimize for "speed of demo" over architectural integrity.

## 1. Zero "Smoke and Mirrors" (No Mocks)
When asked to build features, NEVER use hardcoded arrays, fake JSON responses, or placeholder UI state (e.g., React `useState` mocks) to simulate backend functionality. 
*   If a feature requires a database, build the actual database connection (e.g., Google Cloud Firestore).
*   If it requires a scheduled task, design the actual Async/PubSub or Cloud Scheduler architecture.

## 2. Hard Stops on Missing Dependencies
If a tool requires an API key (e.g., Google Custom Search), library, or system permission that is not present in the environment:
*   **STOP immediately and ask the user how to proceed.** 
*   Do NOT silently write a mock function or return fake results to bypass the error and make the demo "look" successful.

## 3. Strict Plan Adherence
You must rigorously follow the architectural steps outlined in any implementation plan artifacts (e.g., `implementation_plan.md`). Do not skip complex steps like asynchronous queues, robust error handling, or database schemas just to generate a fast output.

## 4. Complete Code Generation (No Laziness)
Never write comments like `// ... rest of the code goes here ...` or `// implement later`. If you are assigned a task, execute the complete, functional logic required for production.

## 5. Destructive File Operations
Never delete untracked files without backing them up first. Always explicitly check for untracked files (`git status`), create a backup, or warn the user before running destructive commands like `git clean -fd` or `git reset --hard`.

## 6. Backend-First Development
Always prioritize building the backend API, database logic, and integrations BEFORE writing frontend code. Never build a UI component without first ensuring the underlying data pipeline or API endpoint actually exists and functions correctly.

## 7. Mandatory Code Audits
Be prepared to act as a rigorous Code Reviewer. If instructed to audit the codebase, you must deeply scan all files for any hardcoded mock data, fake `useState` arrays, or empty placeholder functions, and report them immediately to ensure production integrity.
