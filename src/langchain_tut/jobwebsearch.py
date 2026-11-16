"""Job search agent: Gemini + Tavily web search, returning structured job listings.

Required env vars (in .env): GEMINI_AI_API_KEY, TAVILY_API_KEY.
LangSmith tracing is picked up automatically from the LANGSMITH_* vars.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

TimeRange = Literal["day", "week", "month", "year"]


# ---------- Inputs ----------


@dataclass
class JobSearchCriteria:
    job_title: str
    location: str
    experience: str
    skills: list[str] = field(default_factory=list)
    remote: bool = False
    posted_within: TimeRange = "day"
    max_jobs: int = 10

    def to_query(self) -> str:
        parts = [
            f"Job title: {self.job_title}",
            f"Location: {self.location}" + (" (remote)" if self.remote else ""),
            f"Experience: {self.experience}",
            f"Posted within the last: {self.posted_within}",
            f"Return at most {self.max_jobs} jobs.",
        ]
        if self.skills:
            parts.insert(3, f"Skills: {', '.join(self.skills)}")
        return "\n".join(parts)


# ---------- Structured output ----------


class JobPosting(BaseModel):
    title: str = Field(description="Job title exactly as listed in the posting")
    company: str = Field(description="Hiring company name")
    location: str = Field(description="Location or 'Remote' as listed")
    url: str = Field(description="Direct URL of the posting, taken from search results")
    posted: str | None = Field(
        default=None, description="When it was posted, if the source says so"
    )
    description: str = Field(description="1-2 sentence summary of the role")
    matching_skills: list[str] = Field(
        default_factory=list,
        description="Requested skills that the posting explicitly mentions",
    )


class JobSearchResults(BaseModel):
    jobs: list[JobPosting] = Field(description="Job postings found in search results")
    notes: str | None = Field(
        default=None,
        description="Caveats, e.g. few results found or posting dates unverified",
    )


# ---------- Agent ----------

SYSTEM_PROMPT = """You are a job search assistant with a web search tool.

How to work:
- Run several targeted searches (vary phrasing, e.g. "<title> <location> hiring",
  "<title> jobs <skills>") and prefer job boards and company career pages
  (LinkedIn, Indeed, Naukri, Wellfound, Glassdoor, company sites).
- Only report jobs that actually appear in the search results. Never invent
  companies, titles, or URLs. Every job must have a URL from the results.
- Skip aggregator/listing pages that are not a single specific job when possible.
- Drop postings that clearly don't match the requested role, experience or location.
- Skills are preferences, not hard filters: include a posting if it matches the
  role and some of the skills, and list which ones in `matching_skills`.
  Rank jobs that match more skills first.
- If you can't confirm a posting date, leave `posted` empty and mention it in notes.
- If you find fewer jobs than requested, return what you found and explain in notes.
"""


def build_agent(posted_within: TimeRange = "day"):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,  # factual extraction, not creative writing
        google_api_key=os.getenv("GEMINI_AI_API_KEY"),
    )
    search = TavilySearch(
        max_results=8,
        topic="general",
        time_range=posted_within,
        search_depth="advanced",
    )
    return create_agent(
        model=llm,
        tools=[search],
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(JobSearchResults),
    )


def search_jobs(criteria: JobSearchCriteria) -> JobSearchResults:
    agent = build_agent(criteria.posted_within)
    state = agent.invoke(
        {"messages": [{"role": "user", "content": criteria.to_query()}]},
        config={
            "run_name": "job-web-search",  # name shown in LangSmith
            "tags": ["job-search"],
            "metadata": {"job_title": criteria.job_title, "location": criteria.location},
            "recursion_limit": 25,  # cap on agent steps
        },
    )
    return state["structured_response"]


def print_results(results: JobSearchResults) -> None:
    if not results.jobs:
        print("No matching jobs found.")
    for i, job in enumerate(results.jobs, 1):
        print(f"\n{i}. {job.title} — {job.company}")
        print(f"   Location: {job.location}")
        if job.posted:
            print(f"   Posted:   {job.posted}")
        if job.matching_skills:
            print(f"   Skills:   {', '.join(job.matching_skills)}")
        print(f"   {job.description}")
        print(f"   {job.url}")
    if results.notes:
        print(f"\nNotes: {results.notes}")


def main():
    criteria = JobSearchCriteria(
        job_title="Software Engineer (Backend)",
        location="India",
        remote=True,
        experience="2-3 years",
        skills=["AWS", "Python", "JavaScript", "Java"],
        posted_within="day",
    )
    print_results(search_jobs(criteria))


if __name__ == "__main__":
    main()
