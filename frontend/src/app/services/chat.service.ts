import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, map } from 'rxjs';

import { BaseService } from './base.service';

/**
 * One supporting evidence row returned alongside a chat answer. Rows are
 * keyed either by a single `id` (e.g. game_details) or by a composite
 * `game_id` + `person_id` (e.g. player_box_scores), so both are optional.
 */
export interface Evidence {
  table: string;
  id?: number | string;
  game_id?: number | string;
  person_id?: number | string;
  doc?: string;
  score?: number;
}

/** Shape of the POST /api/chat response body. */
export interface ChatResponse {
  answer: string;
  evidence?: Evidence[];
}

/** One recorded question and response in assets/demo.json. */
export interface DemoEntry extends ChatResponse {
  question: string;
}

/** True when served anywhere except a local dev machine. */
export const DEMO_MODE =
  window.location.search.includes('demo=1') ||
  !['localhost', '127.0.0.1'].includes(window.location.hostname);
@Injectable({
  providedIn: 'root'
})
export class ChatService extends BaseService {
  constructor(protected override http: HttpClient) {
    super(http);
  }

  /** Loads the recorded sample questions used in demo mode. */
  loadDemo(): Observable<DemoEntry[]> {
    return this.http.get<DemoEntry[]>('assets/demo.json');
  }

  /**
   * Answers a question from the live backend, or from recorded responses in demo mode.
   * @param question the user's natural-language question
   * @return an Observable of the answer and supporting evidence
   */
  sendMessage(question: string): Observable<ChatResponse> {
    if (DEMO_MODE) {
      return this.loadDemo().pipe(
        map((entries) => {
          const hit = entries.find((e) => e.question === question.trim());
          return hit ?? { answer: 'Demo mode: please pick one of the sample questions.', evidence: [] };
        })
      );
    }
    const endpoint = `${this.baseUrl}/chat`;
    return this.post<ChatResponse>(endpoint, { question });
  }
}